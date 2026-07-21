"""Validation tests for the MEFS matched-field / nulling framework.

Run with ``pytest tests/test_mefs_framework.py`` or ``python tests/test_mefs_framework.py``.
Uses a 128² grid so the full suite runs in well under a minute.
"""

import numpy as np
import hcipy as hc
import pytest

from PLsim.otf import (
    CoronagraphOTF,
    coupling_efficiency,
    cosine_similarity,
    point_null_port,
    disk_quadrature,
    hard_null,
    soft_null,
    photon_snr,
    max_snr_port,
)

NDIM = 128
DIAMETER = 10.0
WAVELENGTH = 1.55e-6
LOD = WAVELENGTH / DIAMETER


def _build(chain_kind, rcore=10e-6, orthonormalize=True):
    pupil_grid = hc.make_pupil_grid(NDIM, DIAMETER)
    focal_grid = hc.make_pupil_grid(NDIM, 1.0e-6 * NDIM)
    aperture = hc.make_circular_aperture(DIAMETER)(pupil_grid)

    if chain_kind == "vortex":
        chain = [hc.VortexCoronagraph(pupil_grid, charge=2),
                 hc.Apodizer(hc.make_circular_aperture(DIAMETER * 0.95)(pupil_grid))]
    elif chain_kind == "perfect":
        chain = [hc.PerfectCoronagraph(aperture)]
    else:
        raise ValueError(chain_kind)

    otf = CoronagraphOTF.from_fiber_params(
        1.444, 1.444 - 5.5e-3, rcore, WAVELENGTH,
        focal_grid, pupil_grid, DIAMETER * 5,
        aperture, coronagraph_chain=chain, optimize_focal_length=True,
    )
    if orthonormalize:
        otf.orthonormalize_pupil_modes()
    return otf


@pytest.fixture(scope="module")
def vortex_otf():
    return _build("vortex")


@pytest.fixture(scope="module")
def perfect_otf():
    return _build("perfect")


def test_orthonormalization(vortex_otf):
    G = vortex_otf.mode_gram()
    assert np.max(np.abs(G - np.eye(vortex_otf.nmodes))) < 1e-10


def test_adjoint_identity_vortex(vortex_otf):
    _check_adjoint(vortex_otf)


def test_adjoint_identity_perfect(perfect_otf):
    _check_adjoint(perfect_otf)


def _check_adjoint(otf):
    """<phi, T psi> == <T† phi, psi> for random fields, with T† via .backward."""
    rng = np.random.default_rng(0)
    npix = otf.pupil_modes.shape[1]
    phi = rng.standard_normal(npix) + 1j * rng.standard_normal(npix)
    psi = (rng.standard_normal(npix) + 1j * rng.standard_normal(npix)) * np.asarray(otf.aperture)

    def fwd(x):
        wf = hc.Wavefront(hc.Field(x.copy(), otf.pupil_grid), otf.wavelength)
        for el in otf.chain:
            wf = el(wf)
        return np.asarray(wf.electric_field)

    def adj(x):
        wf = hc.Wavefront(hc.Field(x.copy(), otf.pupil_grid), otf.wavelength)
        for el in reversed(otf.chain):
            wf = el.backward(wf)
        return np.asarray(wf.electric_field)

    lhs = np.vdot(phi, fwd(psi))
    rhs = np.vdot(adj(phi), psi)
    assert abs(lhs - rhs) / abs(lhs) < 1e-12


def test_adjoint_coupling_matches_direct(vortex_otf):
    """coupling_at (N adjoint props, cached) == _coupling_for_point (1 forward prop per θ)."""
    offset = (1.0 * LOD, 0.0)
    thetas = np.array([[1.0 * LOD, 0.0],
                       [0.3 * LOD, -0.7 * LOD],
                       [0.0, 0.0]])
    C_fast = vortex_otf.coupling_at(thetas, offset)
    for k, (ax, ay) in enumerate(thetas):
        c_direct = vortex_otf._coupling_for_point(ax, ay, offset)
        scale = max(np.max(np.abs(c_direct)), 1e-30)
        assert np.max(np.abs(C_fast[k] - c_direct)) / scale < 1e-10


def test_quadrature_vs_analytic_coherence(vortex_otf):
    """Dense disk quadrature converges to the analytic J_disk coherence matrix."""
    offset = (1.0 * LOD, 0.0)
    radius = 0.05 * LOD

    R_quad = vortex_otf.coherence_disk(radius, lantern_offset=offset, n_r=24, n_theta=48)

    eff_otf = vortex_otf.to_effective_otf(offset)
    eff_otf.compute()
    R_ana = vortex_otf.coherence_disk_analytic(radius, lantern_offset=offset,
                                               _effective_otf=eff_otf)

    scale = np.max(np.abs(R_quad))
    assert np.max(np.abs(R_quad - R_ana)) / scale < 1e-3
    # Hermitian PSD sanity
    assert np.max(np.abs(R_quad - R_quad.conj().T)) / scale < 1e-12
    assert np.min(np.linalg.eigvalsh(R_quad)) > -1e-12 * scale


def test_point_star_limit(vortex_otf):
    """hard_null with rank-1 R★ at m=1 reproduces the closed-form null tax."""
    offset = (1.0 * LOD, 0.0)
    c_p = vortex_otf.coupling_at(np.array([1.0 * LOD, 0.0]), offset)
    c_s = vortex_otf.coupling_at(np.array([0.01 * LOD, 0.0]), offset)
    P = vortex_otf.incident_power

    eta_free = np.sum(np.abs(c_p) ** 2) / P
    rho = cosine_similarity(c_s, c_p)
    eta_formula = eta_free * (1 - rho ** 2)

    R1 = np.outer(c_s, c_s.conj())
    res = hard_null(R1, c_p, P)
    assert abs(res.eta_planet[0] - eta_free) / eta_free < 1e-12
    assert abs(res.eta_planet[1] - eta_formula) / eta_free < 1e-10

    # explicit Gram–Schmidt port agrees, and the null is exact
    w_gs = point_null_port(c_s, c_p)
    eta_gs = coupling_efficiency(w_gs, c_p, P)
    assert abs(eta_gs - eta_formula) / eta_free < 1e-10
    assert abs(np.vdot(w_gs, c_s)) ** 2 / P < 1e-25 * eta_free

    # m = N removes everything: exactly zero throughput, for any N
    assert res.eta_planet[-1] == 0.0


def test_soft_null_limits(vortex_otf):
    """σ²→∞ recovers the matched filter; σ²→0 approaches the hard null."""
    offset = (1.0 * LOD, 0.0)
    c_p = vortex_otf.coupling_at(np.array([1.0 * LOD, 0.0]), offset)
    P = vortex_otf.incident_power
    R = vortex_otf.coherence_disk(0.05 * LOD, lantern_offset=offset, n_r=8, n_theta=16)
    lam_max = np.max(np.linalg.eigvalsh(R))

    eta_free = np.sum(np.abs(c_p) ** 2) / P
    _, eta_p, eta_s = soft_null(R, c_p, P, np.array([1e-10 * lam_max, 1e8 * lam_max]))
    assert abs(eta_p[1] - eta_free) / eta_free < 1e-6      # matched-filter limit
    assert eta_p[0] < eta_p[1]                              # aggressive null costs throughput
    assert eta_s[0] < eta_s[1]                              # ...and buys suppression

    # frontier monotonicity: increasing σ² never decreases planet throughput
    sig = np.geomspace(1e-10 * lam_max, 1e8 * lam_max, 40)
    _, eta_p_sweep, _ = soft_null(R, c_p, P, sig)
    assert np.all(np.diff(eta_p_sweep) > -1e-12)


def test_max_snr_port(vortex_otf):
    """max_snr_port beats (or ties) both the matched filter and every hard null."""
    offset = (1.0 * LOD, 0.0)
    c_p = vortex_otf.coupling_at(np.array([1.0 * LOD, 0.0]), offset)
    P = vortex_otf.incident_power
    R = vortex_otf.coherence_disk(0.05 * LOD, lantern_offset=offset, n_r=8, n_theta=16)

    contrast, n_star = 1e-5, 1e10
    res = max_snr_port(R, c_p, P, contrast, n_star)

    eta_free = np.sum(np.abs(c_p) ** 2) / P
    snr_matched = photon_snr(eta_free, star_leakage(c_p, R, P), contrast, n_star)
    hard = hard_null(R, c_p, P)
    snr_hard = photon_snr(hard.eta_planet, hard.eta_star, contrast, n_star)

    assert res.snr >= float(snr_matched) - 1e-12
    assert res.snr >= float(np.max(snr_hard)) - 1e-12
    assert abs(np.linalg.norm(res.port) - 1.0) < 1e-12


def test_disk_quadrature_basics():
    pts, wts = disk_quadrature(0.0)
    assert pts.shape == (1, 2) and wts.sum() == 1.0
    pts, wts = disk_quadrature(1e-7, center=(3e-7, -1e-7), n_r=5, n_theta=9)
    assert abs(wts.sum() - 1.0) < 1e-14
    # centroid of a uniform disk is its center
    centroid = (pts * wts[:, None]).sum(axis=0)
    assert np.allclose(centroid, [3e-7, -1e-7], atol=1e-13)
    # mean r² of a uniform disk of radius a is a²/2
    r2 = ((pts - centroid) ** 2).sum(axis=1)
    assert abs((r2 * wts).sum() - 0.5e-14) / 0.5e-14 < 0.02


# needed by test_max_snr_port
from PLsim.otf import star_leakage  # noqa: E402


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
