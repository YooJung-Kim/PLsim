"""
Diagnostic plotting and scanning utilities for PLsim Device objects.

These functions take pre-computed projection grids (``out_grid``) so that
expensive SceneProjector calls are done once and reused.

Typical usage::

    from PLsim.utils.diagnostics import plot_maps, scan

    fov = 100e-3 / 206265
    ngrid = 15
    out_grid = scene_projector.compute_point_grid(fov, ngrid)
    out_grid_x = out_grid[:, :, ngrid // 2, :].copy()

    plot_maps(device, out_grid, fov, labels=labels)

    t = scan(device, out_grid_x, 'lp11amzi', np.linspace(0, 2 * np.pi, 30))
    plt.imshow(t[:, :, 7].T, cmap='turbo')
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors


def plot_maps(
    device,
    out_grid: np.ndarray,
    fov: float,
    labels: list[str] | None = None,
    return_results: bool = False,
    ncols: int = 5,
    figsize: tuple[float, float] | None = None,
    plot_scale: str = 'linear',
    return_fig: bool = False,
) -> np.ndarray | None:
    """Plot a 2-D intensity map for each output port of a device.

    Parameters
    ----------
    device:
        A configured ``Device`` whose outputs will be plotted.
    out_grid:
        Pre-computed projection grid from
        ``SceneProjector.compute_point_grid``.
        Shape ``(n_wavelengths, n_modes, n_modes, ny, nx)``.
    fov:
        Full field of view in **radians**, used to label axes in mas.
    labels:
        Human-readable label per output port.  Defaults to
        ``device.pic.output_names``.
    return_results:
        If ``True``, return the raw output array of shape
        ``(n_wavelengths, n_ports, ny, nx)``.
    ncols:
        Number of subplot columns (default 5).
    figsize:
        Override figure size; defaults to ``(ncols * 2, nrows * 2)``.

    Returns
    -------
    np.ndarray or None
        Output array when ``return_results=True``, otherwise ``None``.
    """
    out = device.calculate_outputs(out_grid)  # (nwav, n_ports, ny, nx)
    n_ports = out.shape[1]

    if labels is None:
        labels = device.pic.output_names if device.pic is not None else [str(i) for i in range(n_ports)]

    nrows = int(np.ceil(n_ports / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize or (ncols * 2, nrows * 2))
    axs = np.array(axs).flatten()

    fov_mas = fov * 206265e3
    extent = (-fov_mas / 2, fov_mas / 2, -fov_mas / 2, fov_mas / 2)

    for i in range(n_ports):
        if plot_scale == 'linear':
            p = axs[i].imshow(out[0, i], extent=extent, origin='lower')
        else:
            p = axs[i].imshow(out[0, i], extent=extent, origin='lower', norm=matplotlib.colors.LogNorm())
        plt.colorbar(p, ax=axs[i])
        axs[i].set_title(labels[i] if i < len(labels) else str(i))
        axs[i].set_xlabel('x (mas)')
        axs[i].set_ylabel('y (mas)')
        axs[i].plot(0, 0, '*', color='white', ms=8)

    for j in range(n_ports, len(axs)):
        axs[j].set_visible(False)

    plt.tight_layout()
    if not return_fig:
        plt.show()
    else:
        return fig, axs
    if return_results:
        return out


def scan(
    device,
    out_grid: np.ndarray,
    param_name: str,
    param_values: np.ndarray,
) -> np.ndarray:
    """Sweep one device parameter and collect outputs at each value.

    The device is updated in-place on each iteration. After the scan, the
    device is left at the **last** value in ``param_values``.

    Parameters
    ----------
    device:
        A ``Device`` object to sweep.
    out_grid:
        Pre-computed projection grid (e.g. a 1-D slice
        ``out_grid[:, :, ngrid // 2, :]`` or the full 2-D grid).
        Passed directly to ``device.calculate_outputs``.
    param_name:
        Name of the parameter to vary (must be accepted by
        ``device.update_pic_matrix``).
    param_values:
        1-D array of values to sweep over.

    Returns
    -------
    np.ndarray
        Shape ``(n_values, n_ports, ...)``, where ``...`` matches the
        trailing spatial dimensions of ``out_grid``.

    Examples
    --------
    Sweep ``lp11amzi`` over a 1-D x-cut and display as an image::

        out_grid_x = out_grid[:, :, ngrid // 2, :].copy()
        t = scan(device, out_grid_x, 'lp11amzi', np.linspace(0, 2*np.pi, 30))
        plt.imshow(t[:, :, 7].T, cmap='turbo')
        plt.yticks(np.arange(n_ports), labels=labels)
    """
    results = []
    for val in param_values:
        device.update_pic_matrix(**{param_name: val})
        out = device.calculate_outputs(out_grid)
        results.append(out[0])
    return np.array(results)  # (n_values, n_ports, ...)
