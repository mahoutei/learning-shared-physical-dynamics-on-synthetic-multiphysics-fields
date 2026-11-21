"""Matched GPhyT residual and spectral-correction controls for the Burgers pilot."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from gphyt.counterfactual_burgers import make_episodes, score_pair, score_spectral, summarize
from gphyt.counterfactual_transport import FORECAST_TIMES, GRID, fit_transport, wavevectors
from gphyt.models.transformer.model import PhysicsTransformer

DT = float(FORECAST_TIMES[0])
SUPPORT_TRANSITIONS = 4


def pack_support(transitions, prior):
    shape = transitions[0].initial.shape
    features = np.zeros((SUPPORT_TRANSITIONS, 3, *shape), dtype=np.float32)
    for index, transition in enumerate(transitions):
        features[index, 0] = transition.initial
        features[index, 1] = transition.final
        features[index, 2] = transition.elapsed / DT
    fit = fit_transport(transitions, prior)
    return torch.from_numpy(features.reshape(-1, *shape)), torch.tensor(
        fit.coefficients, dtype=torch.float32,
    )


def pack_features(transitions):
    """Support features without a coefficient fit (for oracle conditioning)."""
    shape = transitions[0].initial.shape
    features = np.zeros((SUPPORT_TRANSITIONS, 3, *shape), dtype=np.float32)
    for index, transition in enumerate(transitions):
        features[index, 0] = transition.initial
        features[index, 1] = transition.final
        features[index, 2] = transition.elapsed / DT
    return torch.from_numpy(features.reshape(-1, *shape))


class SupportStepper(nn.Module):
    """Use identical support inputs, adding the learned update to a different base."""

    def __init__(self, *, hybrid, backbone=None, hidden_dim=48, num_layers=2):
        super().__init__()
        self.hybrid = hybrid
        self.backbone = backbone if backbone is not None else PhysicsTransformer(
            num_fields=1 + 3 * SUPPORT_TRANSITIONS + 3,
            hidden_dim=hidden_dim,
            mlp_dim=2 * hidden_dim,
            num_heads=3,
            num_layers=num_layers,
            patch_size=(1, 8, 8),
            img_size=(1, *GRID),
        )
        kx, ky = wavevectors(GRID)
        self.register_buffer("radius_squared", torch.tensor(kx**2 + ky**2, dtype=torch.float32))
        kx[:, GRID[1] // 2] = 0
        ky[GRID[0] // 2] = 0
        self.register_buffer("kx", torch.tensor(kx, dtype=torch.float32))
        self.register_buffer("ky", torch.tensor(ky, dtype=torch.float32))
        self.register_buffer("fit_scale", torch.tensor([1.0, 1.0, 100.0]))

    def forward(self, state, support, coefficients):
        fit_channels = (coefficients * self.fit_scale)[:, :, None, None]
        fit_channels = fit_channels.expand(-1, -1, *state.shape[-2:])
        inputs = torch.cat((state[:, None], support, fit_channels), dim=1)
        update = self.backbone.differentiate(inputs.movedim(1, -1)[:, None])[:, 0, :, :, 0]
        base = state
        if self.hybrid:
            vx, vy, diffusivity = coefficients[:, :, None, None].unbind(1)
            generator = -diffusivity * self.radius_squared - 1j * (vx * self.kx + vy * self.ky)
            base = torch.fft.ifft2(torch.fft.fft2(state) * torch.exp(DT * generator)).real
        return base + DT * (update - update.mean(dim=(-2, -1), keepdim=True))


def prepare(episodes, prior, device):
    trajectories = []
    contexts = {"restricted": [], "restored": [], "oracle": []}
    for episode in episodes:
        for support, extra, target, coefficients in zip(
            episode.support, episode.additional_support, episode.targets,
            episode.coefficients, strict=True,
        ):
            trajectories.append(np.concatenate((episode.query[None], target)))
            contexts["restricted"].append(pack_support(support, prior))
            restored_features, restored_fit = pack_support((*support, *extra), prior)
            contexts["restored"].append((restored_features, restored_fit))
            # Oracle conditioning: the true linear coefficients of this pair arm.
            # The nonlinear flux strength stays outside the library by construction.
            contexts["oracle"].append(
                (restored_features, torch.tensor(coefficients[:3], dtype=torch.float32)),
            )
    tensors = {
        name: tuple(torch.stack(column).to(device) for column in zip(*values, strict=True))
        for name, values in contexts.items()
    }
    return torch.tensor(np.stack(trajectories), dtype=torch.float32, device=device), tensors


def train(model, trajectories, contexts, seed, steps):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    rng = torch.Generator().manual_seed(seed)
    started = perf_counter()
    for step in range(steps):
        sample = torch.randint(len(trajectories), (8,), generator=rng)
        time = torch.randint(trajectories.shape[1] - 2, (8,), generator=rng)
        condition = "restricted" if step % 4 == 0 else "restored"
        support, coefficients = (tensor[sample] for tensor in contexts[condition])
        state = trajectories[sample, time]
        first = model(state, support, coefficients)
        second = model(first.detach(), support, coefficients)
        loss = ((first - trajectories[sample, time + 1]).square().mean()
                + (second - trajectories[sample, time + 2]).square().mean()) / 2
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    if trajectories.is_cuda:
        torch.cuda.synchronize()
    return {"training_seconds": perf_counter() - started, "last_batch_loss": float(loss.detach())}


@torch.inference_mode()
def evaluate(model, trajectories, contexts):
    model.eval()
    swapped = torch.arange(len(trajectories), device=trajectories.device) ^ 1
    predictions = {}
    for name, (support, coefficients) in {
        **contexts, "swapped": tuple(tensor[swapped] for tensor in contexts["restored"]),
    }.items():
        state = trajectories[:, 0]
        frames = []
        for _ in FORECAST_TIMES:
            state = model(state, support, coefficients)
            frames.append(state.cpu().numpy())
        predictions[name] = np.stack(frames, axis=1)
    return predictions


def run(seeds, train_pairs, test_pairs, steps, device, output, hidden_dim=48, num_layers=2):
    report = {
        "protocol": {"seeds": seeds, "train_pairs_per_seed": train_pairs,
                     "test_pairs_per_seed": test_pairs, "updates": steps, "device": str(device),
                     "hidden_dim": hidden_dim, "num_layers": num_layers},
        "training": [],
    }
    records = {}
    for seed in seeds:
        print(f"Preparing seed {seed}", flush=True)
        training = list(make_episodes("train", seed, train_pairs))
        test = list(make_episodes("test", seed, test_pairs))
        prior = np.mean([c[:3] for episode in training for c in episode.coefficients], axis=0)
        train_data, train_contexts = prepare(training, prior, device)
        test_data, test_contexts = prepare(test, prior, device)
        for index, episode in enumerate(test):
            methods = records.setdefault(episode.intervention, {})
            for name, scores in score_spectral(episode, prior):
                methods.setdefault(f"spectral_{name}", []).append(
                    {"seed": seed, "pair": index, **scores},
                )
        for variant in ("residual", "hybrid"):
            torch.manual_seed(seed)
            model = SupportStepper(hybrid=variant == "hybrid", hidden_dim=hidden_dim,
                                   num_layers=num_layers).to(device)
            info = train(model, train_data, train_contexts, seed, steps)
            predictions = evaluate(model, test_data, test_contexts)
            for condition, values in predictions.items():
                for index, episode in enumerate(test):
                    scores = score_pair(values[2 * index:2 * index + 2], episode.targets)
                    records[episode.intervention].setdefault(f"{variant}_{condition}", []).append(
                        {"seed": seed, "pair": index, **scores},
                    )
            if not all(np.isfinite(p).all() for p in predictions.values()):
                raise ValueError(f"Nonfinite forecasts for {variant} seed {seed}")
            torch.save(model.state_dict(), output / f"{variant}-seed{seed}.pt")
            report["training"].append({"seed": seed, "variant": variant, **info})
            print(f"Completed {variant} seed {seed}: {info}", flush=True)
        report["test"] = {
            intervention: {"summary": summarize(rows), "pairs": rows}
            for intervention, rows in records.items()
        }
        (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False),
                                             encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--train-pairs", type=int, default=32)
    parser.add_argument("--test-pairs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.train_pairs, args.test_pairs) < 2 or args.steps < 1 or min(args.seeds) < 0:
        parser.error("provide at least two pairs per split, positive steps, and nonnegative seeds")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    run(args.seeds, args.train_pairs, args.test_pairs, args.steps, torch.device(args.device),
        args.output, hidden_dim=args.hidden_dim, num_layers=args.num_layers)


if __name__ == "__main__":
    main()
