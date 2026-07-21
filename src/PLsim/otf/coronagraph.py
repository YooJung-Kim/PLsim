"""
Coronagraph + off-axis photonic lantern OTF.

Physics recap
-------------
A point source at sky angle θ = (θx, θy) enters the pupil as a tilted wavefront
A(p) * exp(2πi p·θ / λ).  The coronagraph chain (e.g. vortex + Lyot stop) maps
this to a post-Lyot pupil field E_pupil(p; θ).  The photonic lantern is placed at
focal position (ox, oy) *in radians*, so LP mode i at that offset corresponds to
the shifted pupil mode:

    φ_i_offset(p) = φ_i(p) * exp(2πi p·(ox, oy) / λ)

The complex coupling amplitude is then:

    c_i(θ; offset) = ∫ φ_i_offset*(p) E_pupil(p; θ) dp
                   = ∫ φ_i*(p) exp(-2πi p·offset / λ) E_pupil(p; θ) dp

Typical workflow
----------------
    coro_otf = CoronagraphOTF.from_fiber_params(
        ncore, nclad, rcore, wavelength,
        focal_grid, pupil_grid, focal_length,
        aperture,
        coronagraph_chain=[vortex, lyot_stop],
    )

    # Expensive step — run once per sky grid
    coro_otf.precompute_sky_response(fov=10e-3, ngrid=64)

    # Cheap — run for each lantern offset
    for offset in offsets:
        maps = coro_otf.compute_coupling_maps(lantern_offset=offset)
        C    = coro_otf.compute_coherence(scene_image)
        out  = device.calculate_outputs(C)

The coronagraph_chain must be a list of HCIPy optical elements whose combined
action maps a pupil-plane Wavefront to another pupil-plane Wavefront (i.e. the
chain must end in the pupil plane, e.g. after the Lyot stop apodizer).
"""

from __future__ import annotations

import numpy as np
import hcipy as hc


class CoronagraphOTF:
    """
    OTF for a coronagraph + off-axis photonic lantern system (forward-scan).

    Parameters
    ----------
    pupil_grid : hcipy.CartesianGrid
        Grid on which the pupil is defined (meters).
    pupil_modes : array_like, shape (nmodes, npix)
        LP fiber modes back-propagated to the pupil plane.
    aperture : array_like, shape (npix,)
        Pupil aperture mask (complex).
    wavelength : float
        Reference wavelength (metres).
    coronagraph_chain : list of hcipy optical elements
        Applied in order to a pupil-plane Wavefront. The final element must
        leave the wavefront in the pupil plane (e.g. a Lyot-stop Apodizer).
    focal_modes : list of array_like, optional
        Focal-plane LP mode fields on focal_grid, stored for diagnostics.
    """

    def __init__(
        self,
        pupil_grid: hc.CartesianGrid,
        pupil_modes: np.ndarray,
        aperture,
        wavelength: float,
        coronagraph_chain: list,
        focal_modes=None,
    ):
        self.pupil_grid = pupil_grid
        self.pupil_modes = np.asarray(pupil_modes, dtype=complex)   # (nmodes, npix)
        self.aperture    = np.asarray(aperture, dtype=complex).ravel()
        self.wavelength  = wavelength
        self.chain       = coronagraph_chain
        self.focal_modes = focal_modes
        self.nmodes      = self.pupil_modes.shape[0]

        # weights is a scalar for uniform grids; broadcast to (npix,)
        w = pupil_grid.weights
        npix = self.pupil_modes.shape[1]
        self._weights = np.full(npix, float(w))

        # Set by precompute_sky_response
        self._E_pupils  = None   # (npoints, npix) complex — post-coronagraph pupil fields
        self._sky_fov   = None
        self._sky_ngrid = None

        # Adjoint-propagated modes, keyed by lantern offset (see effective_modes)
        self._effective_modes_cache: dict[tuple[float, float], np.ndarray] = {}

        # Set by compute_coupling_maps / compute
        self.coupling_maps  = None   # (nmodes, ngrid, ngrid) complex
        self.lantern_offset = None

    # ------------------------------------------------------------------
    # Mode basis
    # ------------------------------------------------------------------

    @property
    def incident_power(self) -> float:
        """Power incident on the (pre-coronagraph) telescope aperture.

        A tilted plane wave has unit modulus across the pupil, so this is the
        same for every sky position — the natural normalization for coupling
        efficiencies.
        """
        return float(np.sum(np.abs(self.aperture) ** 2 * self._weights))

    def mode_gram(self) -> np.ndarray:
        """Gram matrix G[i,j] = <φ_i, φ_j> of the pupil modes (weighted inner product)."""
        return (self.pupil_modes.conj() * self._weights[np.newaxis, :]) @ self.pupil_modes.T

    def orthonormalize_pupil_modes(self) -> None:
        """Löwdin-orthonormalize the pupil modes in the weighted inner product.

        Back-propagating focal-plane LP modes to a finite pupil grid loses power
        outside the grid, leaving the pupil-plane basis slightly non-orthonormal
        (e.g. LP01↔LP02 overlap ~1e-2 at 256²). The matched-filter and null-tax
        formulas assume an orthonormal basis, so symmetric (Löwdin) correction
        is applied: modes ← G^{-1/2} modes, the orthonormal basis closest to the
        original in least-squares sense.
        """
        G = self.mode_gram()
        evals, evecs = np.linalg.eigh(G)
        G_inv_sqrt = (evecs / np.sqrt(evals)[np.newaxis, :]) @ evecs.conj().T
        # rows are modes: mode'_j = Σ_k G^{-1/2}[k,j] mode_k  ⇒  M' = G^{-1/2 T} M
        self.pupil_modes = G_inv_sqrt.T @ self.pupil_modes
        self._effective_modes_cache.clear()

    # ------------------------------------------------------------------
    # Adjoint coupling engine
    # ------------------------------------------------------------------
    #
    # c_i(θ; δ) = <φ_i·ramp_δ, T(A·tilt_θ)> = <T†(φ_i·ramp_δ), A·tilt_θ>
    #
    # so after N adjoint propagations per lantern offset δ (cached), the
    # coupling for ANY sky angle θ is a single weighted inner product —
    # no coronagraph propagation per sky point.

    def effective_modes(self, lantern_offset: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
        """Adjoint-propagated modes ψ_i = T†(φ_i · ramp_offset), cached per offset.

        Returns
        -------
        ndarray, shape (nmodes, npix), complex
        """
        key = (float(lantern_offset[0]), float(lantern_offset[1]))
        cached = self._effective_modes_cache.get(key)
        if cached is not None:
            return cached

        px = np.asarray(self.pupil_grid.x)
        py = np.asarray(self.pupil_grid.y)
        ramp = np.exp(2j * np.pi * (px * key[0] + py * key[1]) / self.wavelength)

        psi = np.empty_like(self.pupil_modes)
        for i in range(self.nmodes):
            wf = hc.Wavefront(hc.Field(self.pupil_modes[i] * ramp, self.pupil_grid),
                              self.wavelength)
            for element in reversed(self.chain):
                wf = element.backward(wf)
            psi[i] = np.asarray(wf.electric_field)

        self._effective_modes_cache[key] = psi
        return psi

    def coupling_at(
        self,
        thetas,
        lantern_offset: tuple[float, float] = (0.0, 0.0),
        chunk: int = 256,
    ) -> np.ndarray:
        """Coupling coefficient vectors c(θ) for one or many sky positions.

        Parameters
        ----------
        thetas : array_like, shape (2,) or (K, 2)
            Sky positions (θx, θy) in radians.
        lantern_offset : (float, float)
            Angular position of the lantern centre in radians.
        chunk : int
            Sky positions per matmul block (memory control).

        Returns
        -------
        ndarray, shape (nmodes,) for a single position or (K, nmodes)
        """
        thetas = np.asarray(thetas, dtype=float)
        single = thetas.ndim == 1
        thetas = np.atleast_2d(thetas)

        psi = self.effective_modes(lantern_offset)
        # B[i,p] = ψ_i*(p) · A(p) · w(p): everything except the tilt
        B = psi.conj() * (self.aperture * self._weights)[np.newaxis, :]

        px = np.asarray(self.pupil_grid.x)
        py = np.asarray(self.pupil_grid.y)

        K = thetas.shape[0]
        C = np.empty((K, self.nmodes), dtype=complex)
        for k0 in range(0, K, chunk):
            th = thetas[k0:k0 + chunk]
            tilt = np.exp(2j * np.pi * (px[:, np.newaxis] * th[:, 0][np.newaxis, :]
                                        + py[:, np.newaxis] * th[:, 1][np.newaxis, :])
                          / self.wavelength)                       # (npix, k)
            C[k0:k0 + chunk] = (B @ tilt).T
        return C[0] if single else C

    def coherence_disk(
        self,
        radius: float,
        center: tuple[float, float] = (0.0, 0.0),
        lantern_offset: tuple[float, float] = (0.0, 0.0),
        n_r: int = 16,
        n_theta: int = 32,
    ) -> np.ndarray:
        """Stellar coherence matrix R★ for a uniform disk, by direct quadrature.

        R★ = Σ_k w_k c(θ_k) c(θ_k)† — an incoherent sum of rank-1 outer
        products (never a coherent average of amplitudes). Weights sum to 1 so
        a disk of radius → 0 matches a unit-flux point source.
        """
        from .nulling import disk_quadrature

        points, weights = disk_quadrature(radius, center=center, n_r=n_r, n_theta=n_theta)
        C = self.coupling_at(points, lantern_offset)               # (K, nmodes)
        return (C * weights[:, np.newaxis]).T @ C.conj()

    def to_effective_otf(self, lantern_offset: tuple[float, float] = (0.0, 0.0)):
        """Plain OTF over the adjoint-propagated modes.

        This makes the analytic extended-source machinery (Scene.J_disk /
        J_gauss + SceneProjector) valid behind the coronagraph chain: the
        coupling coefficient is a pure Fourier transform of T†φ_i, so feeding
        ψ_i = T†(φ_i·ramp) into OTF reproduces coronagraphic couplings.

        Call .compute() on the result before using it with SceneProjector.
        Note the OTF cross-correlation is an unweighted pixel sum: its
        coherence output equals coupling_at-based coherence divided by the
        pupil pixel weight squared (and conjugated) — see
        coherence_disk_analytic for the properly normalized wrapper.
        """
        from .core import OTF

        psi = self.effective_modes(lantern_offset)
        return OTF(self.pupil_grid, psi, np.asarray(self.aperture))

    def coherence_disk_analytic(
        self,
        radius: float,
        center: tuple[float, float] = (0.0, 0.0),
        lantern_offset: tuple[float, float] = (0.0, 0.0),
        _effective_otf=None,
    ) -> np.ndarray:
        """Stellar coherence matrix R★ for a uniform disk, via the analytic
        disk-visibility function (no quadrature).

        Equivalent to coherence_disk in the limit of infinitely fine
        quadrature. Pass a precomputed ``to_effective_otf(...)`` (with
        .compute() already called) via ``_effective_otf`` to amortize the
        cross-correlation cost across radii.
        """
        from ..scene.core import Scene, SceneProjector

        otf = _effective_otf
        if otf is None:
            otf = self.to_effective_otf(lantern_offset)
            otf.compute()

        scene = Scene(self.pupil_grid, self.wavelength)
        projector = SceneProjector(otf, scene)
        raw = projector.compute_disk(radius, center[0], center[1])[..., 0]  # (nmodes, nmodes)

        # OTF cross-correlations are unweighted pixel sums, and its coupling
        # convention is the conjugate of ours: R = w² · conj(raw).
        w = float(self.pupil_grid.weights)
        return w ** 2 * raw.conj()

    # ------------------------------------------------------------------
    # Two-phase API  (preferred for offset scans)
    # ------------------------------------------------------------------

    def precompute_sky_response(self, fov: float, ngrid: int) -> None:
        """
        Forward-scan the sky grid and cache the post-coronagraph pupil field
        for each sky position.

        This is the expensive step (one coronagraph propagation per sky point).
        Run it once; then call ``compute_coupling_maps`` for each lantern offset.

        Parameters
        ----------
        fov : float
            Full angular field of view in radians.
        ngrid : int
            Number of grid points along each sky axis (total: ngrid²).
        """
        px = np.asarray(self.pupil_grid.x)
        py = np.asarray(self.pupil_grid.y)
        npix = len(px)

        theta_vec = np.linspace(-fov / 2, fov / 2, ngrid)
        thx, thy  = np.meshgrid(theta_vec, theta_vec)
        thx_flat  = thx.ravel()
        thy_flat  = thy.ravel()
        npoints   = len(thx_flat)

        estimated_gb = npoints * npix * 16 / 1e9
        if estimated_gb > 2.0:
            print(f"Warning: caching E_pupils will use ~{estimated_gb:.1f} GB. "
                  "Consider reducing ngrid.")

        E_pupils = np.empty((npoints, npix), dtype=complex)

        for k in range(npoints):
            tilt  = np.exp(2j * np.pi * (px * thx_flat[k] + py * thy_flat[k]) / self.wavelength)
            field = hc.Field(self.aperture * tilt, self.pupil_grid)
            wf    = hc.Wavefront(field, self.wavelength)
            for element in self.chain:
                wf = element(wf)
            E_pupils[k] = np.asarray(wf.electric_field)

        self._E_pupils  = E_pupils
        self._sky_fov   = fov
        self._sky_ngrid = ngrid

    def compute_coupling_maps(
        self, lantern_offset: tuple[float, float] = (0.0, 0.0)
    ) -> np.ndarray:
        """
        Compute per-mode coupling amplitude maps from the cached sky response.

        Must call ``precompute_sky_response`` first.

        Parameters
        ----------
        lantern_offset : (float, float)
            Angular position of the lantern centre in radians (ox, oy).
            (0, 0) means on-axis.

        Returns
        -------
        coupling_maps : ndarray, shape (nmodes, ngrid, ngrid), complex
            c_i(θ) for each mode i and sky position θ.
        """
        if self._E_pupils is None:
            raise RuntimeError("Call precompute_sky_response() first.")

        ox, oy = lantern_offset
        px     = np.asarray(self.pupil_grid.x)
        py     = np.asarray(self.pupil_grid.y)

        # Conjugate phase ramp for the offset:
        # coupling: phi_i*(p) * exp(-2πi p·offset/λ) · E_pupil(p)
        conj_ramp = np.exp(-2j * np.pi * (px * ox + py * oy) / self.wavelength)

        # effective_modes_conj[i, p] = phi_i*(p) * exp(-2πi p·offset/λ)
        effective_modes_conj = self.pupil_modes.conj() * conj_ramp[np.newaxis, :]  # (nmodes, npix)

        # Weighted inner product for all sky points at once:
        # coupling_flat[i, k] = sum_p effective_modes_conj[i,p] * E_pupils[k,p] * w[p]
        weighted_E = self._E_pupils * self._weights[np.newaxis, :]   # (npoints, npix)
        coupling_flat = effective_modes_conj @ weighted_E.T          # (nmodes, npoints)

        ngrid = self._sky_ngrid
        self.coupling_maps  = coupling_flat.reshape(self.nmodes, ngrid, ngrid)
        self.lantern_offset = lantern_offset
        return self.coupling_maps

    # ------------------------------------------------------------------
    # Convenience all-in-one
    # ------------------------------------------------------------------

    def compute(
        self,
        fov: float,
        ngrid: int,
        lantern_offset: tuple[float, float] = (0.0, 0.0),
    ) -> np.ndarray:
        """
        Forward-scan and compute coupling maps in one call.

        Equivalent to ``precompute_sky_response`` + ``compute_coupling_maps``.
        Use this when scanning a single offset; use the two-phase API when
        scanning many offsets over the same sky grid.

        Returns
        -------
        coupling_maps : ndarray, shape (nmodes, ngrid, ngrid), complex
        """
        self.precompute_sky_response(fov, ngrid)
        return self.compute_coupling_maps(lantern_offset)

    # ------------------------------------------------------------------
    # Coherence matrix from scene
    # ------------------------------------------------------------------

    def compute_coherence(self, scene_image) -> np.ndarray:
        """
        Compute the modal coherence matrix for an incoherent scene.

        C[i,j] = Σ_θ  I(θ) · c_i(θ) · c_j*(θ)

        Parameters
        ----------
        scene_image : array_like, shape (ngrid, ngrid) or (ngrid*ngrid,)
            Sky intensity map.  Must match the grid used in the last
            ``compute`` / ``precompute_sky_response`` call.

        Returns
        -------
        C : ndarray, shape (nmodes, nmodes), complex
            Pass directly to ``Device.calculate_outputs``.
        """
        if self.coupling_maps is None:
            raise RuntimeError("Call compute() or compute_coupling_maps() first.")

        I = np.asarray(scene_image, dtype=float).ravel()           # (ngrid²,)
        c = self.coupling_maps.reshape(self.nmodes, -1)            # (nmodes, ngrid²)
        return (c * I[np.newaxis, :]) @ c.conj().T                 # (nmodes, nmodes)

    def compute_coherence_for_point(
        self, ax: float, ay: float, lantern_offset: tuple[float, float] | None = None
    ) -> np.ndarray:
        """
        Coherence matrix for a single on-sky point source (does not require
        precomputed sky response — useful for quick checks).

        Parameters
        ----------
        ax, ay : float
            Sky position in radians.
        lantern_offset : (float, float), optional
            Overrides the cached offset without updating state.
        """
        offset = lantern_offset if lantern_offset is not None else (
            self.lantern_offset if self.lantern_offset is not None else (0.0, 0.0)
        )
        c = self._coupling_for_point(ax, ay, offset)   # (nmodes,)
        return np.outer(c, c.conj())                   # (nmodes, nmodes)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _coupling_for_point(
        self, ax: float, ay: float, lantern_offset: tuple[float, float]
    ) -> np.ndarray:
        """Coupling amplitudes for one sky point (no caching)."""
        px = np.asarray(self.pupil_grid.x)
        py = np.asarray(self.pupil_grid.y)
        ox, oy = lantern_offset

        conj_ramp            = np.exp(-2j * np.pi * (px * ox + py * oy) / self.wavelength)
        effective_modes_conj = self.pupil_modes.conj() * conj_ramp[np.newaxis, :]

        tilt  = np.exp(2j * np.pi * (px * ax + py * ay) / self.wavelength)
        field = hc.Field(self.aperture * tilt, self.pupil_grid)
        wf    = hc.Wavefront(field, self.wavelength)
        for element in self.chain:
            wf = element(wf)
        E_pupil = np.asarray(wf.electric_field)

        return effective_modes_conj @ (E_pupil * self._weights)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_focal_modes(
        cls,
        focal_modes,
        focal_grid: hc.CartesianGrid,
        pupil_grid: hc.CartesianGrid,
        wavelength: float,
        propagator: hc.FraunhoferPropagator,
        coronagraph_chain: list,
        aperture,
        optimize_focal_length: bool = True,
    ) -> CoronagraphOTF:
        """
        Build from focal-plane LP mode fields.

        Parameters
        ----------
        focal_modes : list of array_like
            Each element is a 1-D complex array on focal_grid.
        propagator : hcipy.FraunhoferPropagator
            Used to back-propagate focal modes to the pupil plane and, if
            ``optimize_focal_length=True``, to find the best focal length.
        optimize_focal_length : bool
            Maximise on-axis coupling efficiency (same logic as OTF.from_focal_modes).
        """
        aperture_arr = np.asarray(aperture, dtype=complex).ravel()

        if optimize_focal_length:
            wf_ap = hc.Wavefront(
                hc.Field(aperture_arr.copy(), pupil_grid), wavelength
            )

            def _neg_eta(fl):
                propagator.focal_length = float(fl[0])
                wf_f = propagator.forward(wf_ap)
                ol   = float(np.real(np.vdot(wf_f.electric_field, wf_f.electric_field))) + 1e-30
                eta  = sum(
                    abs(np.vdot(wf_f.electric_field, np.asarray(m).ravel())) ** 2
                    for m in focal_modes
                ) / ol
                return -eta

            from scipy.optimize import minimize as _minimize
            res = _minimize(_neg_eta, [propagator.focal_length], method='Nelder-Mead')
            propagator.focal_length = float(res.x[0])
            print(f"Optimised focal length: {propagator.focal_length:.3e} m  "
                  f"(η = {-res.fun:.3f})")

        pupil_modes = []
        for mode in focal_modes:
            wf_mode = hc.Wavefront(
                hc.Field(np.asarray(mode).ravel(), focal_grid), wavelength
            )
            pm = np.asarray(propagator.backward(wf_mode).electric_field)
            pupil_modes.append(pm)

        return cls(
            pupil_grid        = pupil_grid,
            pupil_modes       = np.array(pupil_modes),
            aperture          = aperture_arr,
            wavelength        = wavelength,
            coronagraph_chain = coronagraph_chain,
            focal_modes       = focal_modes,
        )

    @classmethod
    def from_fiber_params(
        cls,
        ncore: float,
        nclad: float,
        rcore: float,
        wavelength: float,
        focal_grid: hc.CartesianGrid,
        pupil_grid: hc.CartesianGrid,
        focal_length: float,
        aperture,
        coronagraph_chain: list,
        optimize_focal_length: bool = True,
    ) -> CoronagraphOTF:
        """
        Build from fiber parameters (LP modes computed internally).

        Parameters
        ----------
        ncore, nclad : float
            Core and cladding refractive indices.
        rcore : float
            Core radius (metres).
        focal_length : float
            Initial focal length for the Fraunhofer propagator (metres).
        aperture : array_like
            Pupil aperture mask on pupil_grid.
        coronagraph_chain : list of hcipy optical elements
            Must end with a pupil-plane element (e.g. Lyot Apodizer).
        """
        from PLsim.utils.LPmodes import compute_lpbases

        ndim                  = pupil_grid.shape[0]
        focal_plane_resolution = (
            focal_grid.x[-1] - focal_grid.x[0] + focal_grid.delta[0]
        ) / ndim

        focal_modes, _ = compute_lpbases(
            ncore, nclad, rcore, wavelength, ndim, focal_plane_resolution
        )

        propagator = hc.FraunhoferPropagator(
            pupil_grid, focal_grid, focal_length=focal_length
        )

        return cls.from_focal_modes(
            focal_modes       = focal_modes,
            focal_grid        = focal_grid,
            pupil_grid        = pupil_grid,
            wavelength        = wavelength,
            propagator        = propagator,
            coronagraph_chain = coronagraph_chain,
            aperture          = aperture,
            optimize_focal_length = optimize_focal_length,
        )
