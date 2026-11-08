"""Reduced multi-physics corpus generators.

Each generator returns a tensor of shape ``(n_traj, n_frames, H, W, 5)`` with
channels ordered as the paper's physical fields:

    (pressure, density, temperature, velocity_x, velocity_y)

Fields that a system does not contain are zero, following the paper's
convention (Appendix A.5).

The PDEs are solved with Fourier pseudo-spectral methods on a periodic unit
square.  They are deliberately small, deterministic surrogates for the seven
paper datasets: the goal is to test the architecture and the paper's claims at
reduced scale, not to reproduce the original simulation data.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
from torch import Tensor

FIELD_NAMES = ("pressure", "density", "temperature", "velocity")
N_CHANNELS = 5


# ---------------------------------------------------------------------------
# spectral helpers
# ---------------------------------------------------------------------------


def _freqs(h: int, w: int, device: torch.device) -> tuple[Tensor, Tensor]:
    kx = 2.0 * math.pi * torch.fft.fftfreq(h, d=1.0 / h, device=device)
    ky = 2.0 * math.pi * torch.fft.rfftfreq(w, d=1.0 / w, device=device)
    return kx[:, None], ky[None, :]


def _dealias_mask(h: int, w: int, device: torch.device, fraction: float = 2.0 / 3.0) -> Tensor:
    kx = torch.fft.fftfreq(h, d=1.0, device=device) * h
    ky = torch.fft.rfftfreq(w, d=1.0, device=device) * w
    return (kx.abs() <= fraction * h / 2.0)[:, None] & (
        ky.abs() <= fraction * w / 2.0
    )[None, :]


def _grad(q_hat: Tensor, k: Tensor) -> Tensor:
    """Spectral derivative of a real field given its rfft2 coefficients."""
    return torch.fft.irfft2(1j * k * q_hat)


def _to_hat(q: Tensor, dealias: Tensor | None = None) -> Tensor:
    q_hat = torch.fft.rfft2(q)
    if dealias is not None:
        q_hat = q_hat * dealias
    return q_hat


def _etd_update(
    q_hat: Tensor,
    rhs_hat: Tensor,
    k2: Tensor,
    diffusivity: float,
    dt: float,
) -> Tensor:
    """ETD1 update for dq/dt = diffusivity * laplacian(q) + rhs."""
    lam = -diffusivity * k2
    e = torch.exp(lam * dt)
    # (exp(lam*dt) - 1) / lam, with the lam -> 0 limit equal to dt
    phi = torch.where(lam.abs() > 1e-12, (e - 1.0) / lam, torch.full_like(lam, dt))
    return e * q_hat + phi * rhs_hat


def _random_smooth(
    n: int,
    h: int,
    w: int,
    device: torch.device,
    generator: torch.Generator,
    k0: float,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Band-limited Gaussian random field with unit standard deviation."""
    noise = torch.randn(n, h, w, device=device, generator=generator, dtype=dtype)
    kx, ky = _freqs(h, w, device)
    k2 = kx**2 + ky**2
    filt = torch.exp(-k2 / (2.0 * k0**2))
    q = torch.fft.irfft2(torch.fft.rfft2(noise) * filt, s=(h, w))
    q = q - q.mean(dim=(-2, -1), keepdim=True)
    q = q / (q.std(dim=(-2, -1), keepdim=True) + 1e-8)
    return q


def _poisson_solve(rhs_hat: Tensor, k2: Tensor) -> Tensor:
    safe = torch.where(k2 > 1e-12, k2, torch.ones_like(k2))
    out = torch.where(k2 > 1e-12, rhs_hat / safe, torch.zeros_like(rhs_hat))
    return torch.fft.irfft2(out)





# ---------------------------------------------------------------------------
# systems
# ---------------------------------------------------------------------------


def advection_diffusion(
    n_traj: int,
    n_frames: int,
    h: int,
    w: int,
    seed: int,
    device: torch.device,
) -> Tensor:
    """Passive scalar advected by a steady incompressible flow with diffusion."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    kx, ky = _freqs(h, w, device)
    k2 = kx**2 + ky**2
    dealias = _dealias_mask(h, w, device)

    psi = 0.12 * _random_smooth(n_traj, h, w, device, gen, k0=2.0)
    u = _grad(_to_hat(psi, dealias), ky)
    v = _grad(_to_hat(psi, dealias), -kx)
    s = 0.7 * _random_smooth(n_traj, h, w, device, gen, k0=3.0) + 1.0
    s_hat = torch.fft.rfft2(s)

    dt, record_every, kappa = 2e-3, 10, 2e-3
    frames = torch.empty(n_traj, n_frames, h, w, N_CHANNELS, device=device)
    out = torch.zeros_like(s)

    for i in range(n_frames):
        if i == 0:
            frames[:, 0, ..., 1] = s
        for _ in range(0 if i == 0 else record_every):
            s_x = _grad(s_hat, kx)
            s_y = _grad(s_hat, ky)
            rhs = _to_hat(-(u * s_x + v * s_y), dealias)
            s_hat = _etd_update(s_hat, rhs, k2, kappa, dt)
            s = torch.fft.irfft2(s_hat, s=(h, w))
        if i > 0:
            frames[:, i, ..., 1] = s
    frames[..., 0] = out[:, None]  # pressure
    frames[..., 2] = out[:, None]  # temperature
    frames[..., 3] = out[:, None]  # vx
    frames[..., 4] = out[:, None]  # vy
    return frames


def burgers2d(
    n_traj: int,
    n_frames: int,
    h: int,
    w: int,
    seed: int,
    device: torch.device,
) -> Tensor:
    """Coupled 2D viscous Burgers equations (nonlinear shocks)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    kx, ky = _freqs(h, w, device)
    k2 = kx**2 + ky**2
    dealias = _dealias_mask(h, w, device)

    u = 0.5 * _random_smooth(n_traj, h, w, device, gen, k0=3.0)
    v = 0.5 * _random_smooth(n_traj, h, w, device, gen, k0=3.0)
    u_hat = torch.fft.rfft2(u)
    v_hat = torch.fft.rfft2(v)

    dt, record_every, nu = 1e-3, 10, 2e-3
    frames = torch.zeros(n_traj, n_frames, h, w, N_CHANNELS, device=device)

    for i in range(n_frames):
        if i == 0:
            frames[:, 0, ..., 3] = u
            frames[:, 0, ..., 4] = v
        for _ in range(0 if i == 0 else record_every):
            u_x = _grad(u_hat, kx)
            u_y = _grad(u_hat, ky)
            v_x = _grad(v_hat, kx)
            v_y = _grad(v_hat, ky)
            rhs_u = _to_hat(-(u * u_x + v * u_y), dealias)
            rhs_v = _to_hat(-(u * v_x + v * v_y), dealias)
            u_hat = _etd_update(u_hat, rhs_u, k2, nu, dt)
            v_hat = _etd_update(v_hat, rhs_v, k2, nu, dt)
            u = torch.fft.irfft2(u_hat, s=(h, w))
            v = torch.fft.irfft2(v_hat, s=(h, w))
        if i > 0:
            frames[:, i, ..., 3] = u
            frames[:, i, ..., 4] = v
    return frames


def sine_gordon(
    n_traj: int,
    n_frames: int,
    h: int,
    w: int,
    seed: int,
    device: torch.device,
) -> Tensor:
    """Nonlinear wave equation (sine-Gordon), second-order Strang splitting."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    kx, ky = _freqs(h, w, device)
    k2 = kx**2 + ky**2

    c = 0.15
    u = _random_smooth(n_traj, h, w, device, gen, k0=3.0)
    v = torch.zeros_like(u)
    freq = c * torch.sqrt(k2)
    safe = torch.where(freq > 1e-12, freq, torch.ones_like(freq))

    def nonlinear(u: Tensor, v: Tensor, half_dt: float) -> tuple[Tensor, Tensor]:
        return u, v + half_dt * (-torch.sin(u))

    dt, record_every = 2e-3, 10
    frames = torch.zeros(n_traj, n_frames, h, w, N_CHANNELS, device=device)

    for i in range(n_frames):
        if i == 0:
            frames[:, 0, ..., 0] = u
            frames[:, 0, ..., 3] = v
        for _ in range(0 if i == 0 else record_every):
            u, v = nonlinear(u, v, dt / 2)
            u_hat = torch.fft.rfft2(u)
            v_hat = torch.fft.rfft2(v)
            ang = freq * dt
            cos_a, sin_a = torch.cos(ang), torch.sin(ang)
            u_hat, v_hat = (
                cos_a * u_hat + sin_a * v_hat / safe,
                -safe * sin_a * u_hat + cos_a * v_hat,
            )
            u = torch.fft.irfft2(u_hat, s=(h, w))
            v = torch.fft.irfft2(v_hat, s=(h, w))
            u, v = nonlinear(u, v, dt / 2)
        if i > 0:
            frames[:, i, ..., 0] = u
            frames[:, i, ..., 3] = v
    return frames


def incompressible_ns(
    n_traj: int,
    n_frames: int,
    h: int,
    w: int,
    seed: int,
    device: torch.device,
) -> Tensor:
    """2D incompressible Navier-Stokes in vorticity-streamfunction form."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    kx, ky = _freqs(h, w, device)
    k2 = kx**2 + ky**2
    dealias = _dealias_mask(h, w, device)

    omega = 0.5 * _random_smooth(n_traj, h, w, device, gen, k0=3.5)
    omega_hat = torch.fft.rfft2(omega)
    yy = (torch.arange(w, device=device) + 0.5) / w
    force = 0.1 * torch.sin(4.0 * math.pi * yy)[None, None, :]

    dt, record_every, nu = 2e-3, 20, 2e-3
    frames = torch.zeros(n_traj, n_frames, h, w, N_CHANNELS, device=device)

    for i in range(n_frames):
        if i == 0:
            psi = _poisson_solve(omega_hat, k2)
            psi_hat = torch.fft.rfft2(psi)
            u = _grad(psi_hat, ky)
            v = _grad(psi_hat, -kx)
            p = _pressure(u, v, kx, ky, k2, dealias)
            frames[:, 0, ..., 0] = p
            frames[:, 0, ..., 3] = u
            frames[:, 0, ..., 4] = v
        for _ in range(0 if i == 0 else record_every):
            psi = _poisson_solve(omega_hat, k2)
            psi_hat = torch.fft.rfft2(psi)
            u = _grad(psi_hat, ky)
            v = _grad(psi_hat, -kx)
            omega_x = _grad(omega_hat, kx)
            omega_y = _grad(omega_hat, ky)
            rhs = _to_hat(-(u * omega_x + v * omega_y) + force, dealias)
            omega_hat = _etd_update(omega_hat, rhs, k2, nu, dt)
            omega = torch.fft.irfft2(omega_hat, s=(h, w))
        if i > 0:
            psi = _poisson_solve(omega_hat, k2)
            psi_hat = torch.fft.rfft2(psi)
            u = _grad(psi_hat, ky)
            v = _grad(psi_hat, -kx)
            p = _pressure(u, v, kx, ky, k2, dealias)
            frames[:, i, ..., 0] = p
            frames[:, i, ..., 3] = u
            frames[:, i, ..., 4] = v
    return frames


def _pressure(
    u: Tensor,
    v: Tensor,
    kx: Tensor,
    ky: Tensor,
    k2: Tensor,
    dealias: Tensor,
) -> Tensor:
    uu = _to_hat(u * u, dealias)
    uv = _to_hat(u * v, dealias)
    vv = _to_hat(v * v, dealias)
    rhs = (1j * kx) ** 2 * uu + 2 * (1j * kx) * (1j * ky) * uv + (1j * ky) ** 2 * vv
    safe = torch.where(k2 > 1e-12, k2, torch.ones_like(k2))
    p_hat = torch.where(k2 > 1e-12, rhs / safe, torch.zeros_like(rhs))
    return torch.fft.irfft2(p_hat)


def wave(
    n_traj: int,
    n_frames: int,
    h: int,
    w: int,
    seed: int,
    device: torch.device,
) -> Tensor:
    """Linear wave equation, solved exactly in Fourier space (held-out physics)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    kx, ky = _freqs(h, w, device)
    k2 = kx**2 + ky**2

    c = 0.2
    u = _random_smooth(n_traj, h, w, device, gen, k0=4.0)
    v = torch.zeros_like(u)
    u_hat = torch.fft.rfft2(u)
    v_hat = torch.fft.rfft2(v)

    freq = c * torch.sqrt(k2)
    dt, record_every = 2e-3, 10
    frames = torch.zeros(n_traj, n_frames, h, w, N_CHANNELS, device=device)

    for i in range(n_frames):
        if i == 0:
            frames[:, 0, ..., 0] = u
            frames[:, 0, ..., 3] = v
        for _ in range(0 if i == 0 else record_every):
            ang = freq * dt
            cos_a, sin_a = torch.cos(ang), torch.sin(ang)
            safe = torch.where(freq > 1e-12, freq, torch.ones_like(freq))
            u_new = cos_a * u_hat + sin_a * v_hat / safe
            v_new = -safe * sin_a * u_hat + cos_a * v_hat
            u_hat, v_hat = u_new, v_new
            u = torch.fft.irfft2(u_hat, s=(h, w))
            v = torch.fft.irfft2(v_hat, s=(h, w))
        if i > 0:
            frames[:, i, ..., 0] = u
            frames[:, i, ..., 3] = v
    return frames


def thermal_convection(
    n_traj: int,
    n_frames: int,
    h: int,
    w: int,
    seed: int,
    device: torch.device,
) -> Tensor:
    """2D Boussinesq convection in a periodic domain (held-out physics)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    kx, ky = _freqs(h, w, device)
    k2 = kx**2 + ky**2
    dealias = _dealias_mask(h, w, device)

    yy = (torch.arange(w, device=device) + 0.5) / w
    base = 0.5 - 0.5 * torch.cos(2.0 * math.pi * yy)
    T = base[None, None, :] + 0.02 * _random_smooth(n_traj, h, w, device, gen, k0=5.0)
    omega = 0.05 * _random_smooth(n_traj, h, w, device, gen, k0=4.0)
    T_hat = torch.fft.rfft2(T)
    omega_hat = torch.fft.rfft2(omega)

    dt, record_every = 1e-4, 250
    nu, kappa, ra = 2e-3, 2e-3, 30.0
    frames = torch.zeros(n_traj, n_frames, h, w, N_CHANNELS, device=device)

    for i in range(n_frames):
        if i == 0:
            psi = _poisson_solve(omega_hat, k2)
            psi_hat = torch.fft.rfft2(psi)
            u = _grad(psi_hat, ky)
            v = _grad(psi_hat, -kx)
            p = _pressure(u, v, kx, ky, k2, dealias)
            frames[:, 0, ..., 0] = p
            frames[:, 0, ..., 2] = T
            frames[:, 0, ..., 3] = u
            frames[:, 0, ..., 4] = v
        for _ in range(0 if i == 0 else record_every):
            psi = _poisson_solve(omega_hat, k2)
            psi_hat = torch.fft.rfft2(psi)
            u = _grad(psi_hat, ky)
            v = _grad(psi_hat, -kx)
            omega_x = _grad(omega_hat, kx)
            omega_y = _grad(omega_hat, ky)
            T_x = _grad(T_hat, kx)
            T_y = _grad(T_hat, ky)
            rhs_w = _to_hat(-(u * omega_x + v * omega_y) + ra * T_x, dealias)
            rhs_t = _to_hat(-(u * T_x + v * T_y), dealias)
            omega_hat = _etd_update(omega_hat, rhs_w, k2, nu, dt)
            T_hat = _etd_update(T_hat, rhs_t, k2, kappa, dt)
            omega = torch.fft.irfft2(omega_hat, s=(h, w))
            T = torch.fft.irfft2(T_hat, s=(h, w))
        if i > 0:
            psi = _poisson_solve(omega_hat, k2)
            psi_hat = torch.fft.rfft2(psi)
            u = _grad(psi_hat, ky)
            v = _grad(psi_hat, -kx)
            p = _pressure(u, v, kx, ky, k2, dealias)
            frames[:, i, ..., 0] = p
            frames[:, i, ..., 2] = T
            frames[:, i, ..., 3] = u
            frames[:, i, ..., 4] = v
    return frames


GENERATORS: dict[str, Callable[..., Tensor]] = {
    "advection": advection_diffusion,
    "burgers": burgers2d,
    "sine_gordon": sine_gordon,
    "incompressible_ns": incompressible_ns,
    "wave": wave,
    "thermal_convection": thermal_convection,
}

TRAIN_SYSTEMS = ("advection", "burgers", "incompressible_ns", "sine_gordon")
HELDOUT_SYSTEMS = ("wave", "thermal_convection")


def generate(
    name: str,
    n_traj: int,
    n_frames: int,
    h: int,
    w: int,
    seed: int,
    device: torch.device | str = "cpu",
) -> Tensor:
    if name not in GENERATORS:
        raise KeyError(f"Unknown system {name!r}; choose from {sorted(GENERATORS)}")
    device = torch.device(device)
    data = GENERATORS[name](n_traj, n_frames, h, w, seed, device)
    if not torch.isfinite(data).all():
        raise FloatingPointError(f"{name} produced non-finite values")
    return data.float().cpu()
