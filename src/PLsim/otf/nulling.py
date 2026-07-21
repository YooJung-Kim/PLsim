"""
Matched-field coupling and null-constrained starlight rejection.

Pure linear algebra on mode-coupling vectors/matrices — no hcipy dependency.
The physical inputs are produced by ``CoronagraphOTF.coupling_at`` /
``coherence_disk``; everything here operates on:

- ``c_planet`` : (N,) complex — the planet's coupling coefficient vector
- ``c_star``   : (N,) complex — a point star's coupling coefficient vector
- ``R_star``   : (N, N) complex Hermitian PSD — a resolved (incoherent) star's
  coherence matrix, R★ = Σ_k w_k c(θ_k) c(θ_k)†
- ``incident_power`` : float — pre-coronagraph aperture power, the
  normalization that turns |amplitude|² into a coupling *efficiency*

Conventions
-----------
Port amplitude for port vector w is <w, c> = Σ_i conj(w_i) c_i.  The
unconstrained matched-filter port is w★ = c_planet (NOT its conjugate).
All ports are defined up to complex scale; efficiencies always divide by ‖w‖².
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Basic quantities
# ---------------------------------------------------------------------------

def coupling_efficiency(w: np.ndarray, c: np.ndarray, incident_power: float) -> float:
    """η = |<w, c>|² / (‖w‖² · P_inc) for port w and coupling vector c."""
    norm_sq = float(np.sum(np.abs(w) ** 2))
    return float(np.abs(np.vdot(w, c)) ** 2 / (norm_sq * incident_power))


def matched_port(c_planet: np.ndarray) -> np.ndarray:
    """Unconstrained matched-filter port: w★ = c_planet (Cauchy–Schwarz optimum).

    Achieves η_free = ‖c_planet‖² / P_inc.
    """
    return np.asarray(c_planet, dtype=complex).copy()


def cosine_similarity(c_star: np.ndarray, c_planet: np.ndarray) -> float:
    """ρ = |<c_star, c_planet>| / (‖c_star‖·‖c_planet‖) ∈ [0, 1]."""
    return float(
        np.abs(np.vdot(c_star, c_planet))
        / (np.linalg.norm(c_star) * np.linalg.norm(c_planet))
    )


def point_null_port(c_star: np.ndarray, c_planet: np.ndarray) -> np.ndarray:
    """Port maximizing planet throughput subject to exactly nulling a point star.

    Gram–Schmidt projection of c_planet orthogonal to c_star:
        w★ = c_planet − (<c_star, c_planet>/‖c_star‖²) c_star

    Achieves η_nulled = η_free · (1 − ρ²)  (the "null tax").
    """
    c_star = np.asarray(c_star, dtype=complex)
    c_planet = np.asarray(c_planet, dtype=complex)
    return c_planet - (np.vdot(c_star, c_planet) / np.sum(np.abs(c_star) ** 2)) * c_star


# ---------------------------------------------------------------------------
# Resolved star: quadrature and coherence matrix
# ---------------------------------------------------------------------------

def disk_quadrature(
    radius: float,
    center: tuple[float, float] = (0.0, 0.0),
    n_r: int = 16,
    n_theta: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Quadrature points and weights for a uniform disk of angular radius ``radius``.

    Equal-area rings (midpoint rule in r²) crossed with uniform angles.
    Weights sum to 1, so the disk's total flux matches a unit-flux point
    source as radius → 0 (a zero/negative radius returns the single point
    ``center`` with weight 1).

    Returns
    -------
    points : (K, 2) ndarray of (θx, θy) in radians
    weights : (K,) ndarray summing to 1
    """
    cx, cy = center
    if radius <= 0:
        return np.array([[cx, cy]]), np.array([1.0])
    r = radius * np.sqrt((np.arange(n_r) + 0.5) / n_r)
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    R, TH = np.meshgrid(r, theta, indexing="ij")
    points = np.column_stack([cx + (R * np.cos(TH)).ravel(),
                              cy + (R * np.sin(TH)).ravel()])
    weights = np.full(points.shape[0], 1.0 / (n_r * n_theta))
    return points, weights


def coherence_from_couplings(C: np.ndarray, weights=None) -> np.ndarray:
    """R = Σ_k w_k c_k c_k† from stacked coupling vectors C (K, N).

    An incoherent sum of rank-1 outer products — never average the complex
    amplitudes first (that would treat the source as spatially coherent).
    """
    C = np.atleast_2d(np.asarray(C, dtype=complex))
    if weights is None:
        weights = np.full(C.shape[0], 1.0 / C.shape[0])
    return (C * np.asarray(weights)[:, np.newaxis]).T @ C.conj()


def star_leakage(w: np.ndarray, R_star: np.ndarray, incident_power: float) -> float:
    """Residual stellar power fraction through normalized port w:
    L★ = w† R★ w / (‖w‖² · P_inc)."""
    norm_sq = float(np.sum(np.abs(w) ** 2))
    if norm_sq == 0.0:
        return 0.0
    return float(np.real(np.vdot(w, R_star @ w)) / (norm_sq * incident_power))


# ---------------------------------------------------------------------------
# Hard null (fixed number of removed stellar modes)
# ---------------------------------------------------------------------------

@dataclass
class HardNullResult:
    """Hard-null solutions for every number of removed stellar modes m = 0..N.

    Attributes
    ----------
    ports : (N+1, N) complex — unnormalized port vector per m (row m = 0..N)
    eta_planet : (N+1,) — planet throughput of the normalized port per m
    eta_star : (N+1,) — residual stellar leakage L★(m) through the same port
    eigenvalues : (N,) — eigenvalues of R★, descending
    eigenvectors : (N, N) — corresponding eigenvectors as columns, descending
    """

    ports: np.ndarray
    eta_planet: np.ndarray
    eta_star: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray


def hard_null(R_star: np.ndarray, c_planet: np.ndarray, incident_power: float) -> HardNullResult:
    """Project c_planet off the m dominant eigenmodes of R★, for all m = 0..N.

    w(m) = (I − U_m U_m†) c_planet with U_m the m largest-eigenvalue
    directions.  Well-posed for every N including N=1: m = N always gives
    exactly w = 0 (no epsilon or threshold involved).  Removing only the top
    m eigenmodes does NOT guarantee negligible overlap with the remaining
    stellar directions, so the residual leakage eta_star is evaluated
    explicitly for each m rather than assumed small.
    """
    c_planet = np.asarray(c_planet, dtype=complex)
    N = c_planet.shape[0]

    eigvals, eigvecs = np.linalg.eigh(R_star)          # ascending
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]                          # descending, columns

    ports = np.empty((N + 1, N), dtype=complex)
    eta_planet = np.empty(N + 1)
    eta_star = np.empty(N + 1)

    overlaps = eigvecs.conj().T @ c_planet              # <u_k, c_p>, descending
    for m in range(N + 1):
        if m == N:
            # U_N spans the whole space: w = 0 by construction, so return the
            # exact zero rather than the ~machine-epsilon projection residual
            w = np.zeros(N, dtype=complex)
        else:
            w = c_planet - eigvecs[:, :m] @ overlaps[:m]
        ports[m] = w
        eta_planet[m] = np.sum(np.abs(w) ** 2) / incident_power
        eta_star[m] = star_leakage(w, R_star, incident_power)

    return HardNullResult(ports, eta_planet, eta_star, eigvals, eigvecs)


# ---------------------------------------------------------------------------
# Soft null (Tikhonov) and photon-noise SNR
# ---------------------------------------------------------------------------

def soft_null(
    R_star: np.ndarray,
    c_planet: np.ndarray,
    incident_power: float,
    sigma2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tikhonov-regularized ports w(σ²) ∝ (R★ + σ²I)⁻¹ c_planet.

    Sweeping σ² traces the continuous Pareto frontier between aggressive
    stellar suppression (σ² → 0) and the unconstrained matched filter
    (σ² → ∞).

    Parameters
    ----------
    sigma2 : float or (S,) array of regularization scales.

    Returns
    -------
    ports : (S, N) unit-norm port vectors
    eta_planet : (S,) planet throughput
    eta_star : (S,) stellar leakage through the same ports
    """
    c_planet = np.asarray(c_planet, dtype=complex)
    N = c_planet.shape[0]
    sigma2_arr = np.atleast_1d(np.asarray(sigma2, dtype=float))

    ports = np.empty((sigma2_arr.size, N), dtype=complex)
    eta_planet = np.empty(sigma2_arr.size)
    eta_star = np.empty(sigma2_arr.size)
    for i, s2 in enumerate(sigma2_arr):
        w = np.linalg.solve(R_star + s2 * np.eye(N), c_planet)
        w /= np.linalg.norm(w) + 1e-300
        ports[i] = w
        eta_planet[i] = np.abs(np.vdot(w, c_planet)) ** 2 / incident_power
        eta_star[i] = star_leakage(w, R_star, incident_power)
    return ports, eta_planet, eta_star


def photon_snr(eta_planet, eta_star, contrast: float, star_photons: float):
    """Photon-noise-limited detection S/N of the planet through one port.

    S = N★ · contrast · η_planet   (planet counts)
    B = N★ · η_star                (leaked star counts)
    SNR = S / sqrt(S + B)

    ``contrast`` is the planet/star flux ratio; ``star_photons`` (N★) is the
    total photon budget the star delivers to the telescope aperture over the
    exposure.  SNR scales as sqrt(N★), so only relative comparisons matter
    unless N★ is physically motivated.
    """
    S = star_photons * contrast * np.asarray(eta_planet, dtype=float)
    B = star_photons * np.asarray(eta_star, dtype=float)
    total = S + B
    return np.where(total > 0, S / np.sqrt(np.where(total > 0, total, 1.0)), 0.0)


@dataclass
class MaxSNRResult:
    """Best photon-noise SNR port over the achievable trade-off frontier.

    Attributes
    ----------
    port : (N,) unit-norm port vector achieving the best SNR
    snr : float — the best SNR
    eta_planet, eta_star : float — throughput/leakage at the optimum
    sigma2 : float or None — Tikhonov scale of the optimum (None if a
        hard-null corner won)
    frontier_sigma2, frontier_eta_planet, frontier_eta_star, frontier_snr :
        the swept Tikhonov frontier, for plotting
    """

    port: np.ndarray
    snr: float
    eta_planet: float
    eta_star: float
    sigma2: float | None
    frontier_sigma2: np.ndarray
    frontier_eta_planet: np.ndarray
    frontier_eta_star: np.ndarray
    frontier_snr: np.ndarray


def max_snr_port(
    R_star: np.ndarray,
    c_planet: np.ndarray,
    incident_power: float,
    contrast: float,
    star_photons: float,
    n_sigma: int = 120,
) -> MaxSNRResult:
    """Port maximizing photon-noise SNR (allows some starlight leakage).

    The SNR is monotone increasing in η_planet and decreasing in η_star, so
    the optimum lies on the Pareto frontier traced by the Tikhonov family;
    this sweeps σ² across [1e-12, 1e6]·λ_max(R★) and also checks the hard-null
    corner points.  If R★ ≈ 0 (nothing to null) the matched filter is returned.
    """
    c_planet = np.asarray(c_planet, dtype=complex)
    lam_max = float(np.max(np.linalg.eigvalsh(R_star)))

    if lam_max <= 0:
        w = c_planet / np.linalg.norm(c_planet)
        eta_p = float(np.sum(np.abs(c_planet) ** 2) / incident_power)
        snr = float(photon_snr(eta_p, 0.0, contrast, star_photons))
        empty = np.array([])
        return MaxSNRResult(w, snr, eta_p, 0.0, None, empty, empty, empty, empty)

    sigma2_grid = np.geomspace(1e-12 * lam_max, 1e6 * lam_max, n_sigma)
    ports, eta_p, eta_s = soft_null(R_star, c_planet, incident_power, sigma2_grid)
    snrs = photon_snr(eta_p, eta_s, contrast, star_photons)

    best = int(np.argmax(snrs))
    best_port = ports[best]
    best_snr = float(snrs[best])
    best_eta_p, best_eta_s = float(eta_p[best]), float(eta_s[best])
    best_sigma2 = float(sigma2_grid[best])

    hard = hard_null(R_star, c_planet, incident_power)
    hard_snrs = photon_snr(hard.eta_planet, hard.eta_star, contrast, star_photons)
    m_best = int(np.argmax(hard_snrs))
    if hard_snrs[m_best] > best_snr:
        w = hard.ports[m_best]
        best_port = w / (np.linalg.norm(w) + 1e-300)
        best_snr = float(hard_snrs[m_best])
        best_eta_p = float(hard.eta_planet[m_best])
        best_eta_s = float(hard.eta_star[m_best])
        best_sigma2 = None

    return MaxSNRResult(best_port, best_snr, best_eta_p, best_eta_s, best_sigma2,
                        sigma2_grid, eta_p, eta_s, snrs)
