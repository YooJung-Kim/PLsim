"""
PLsim chip design experiment framework.

Provides a structured way to define chip architectures, lantern configurations,
and multi-stage optimization experiments with user-defined loss functions.

Typical usage::

    from PLsim.system.experiment import (
        ChipDesign, LanternConfig, OptimizationStage, Experiment, compare_experiments
    )

    chip = ChipDesign(
        name='my_chip',
        device_template=create_my_pic,
        parameter_names=['ps1', 'ps2', ...],
        initial_parameters={'ps1': 0.0, 'ps2': np.pi, ...},
        port_labels={'a': 'null', 'b': 'antinull'},
    )

    lantern = LanternConfig(
        name='SPL',
        matrix=generate_unitary_matrix_near_identity(3, epsilon=0.3)[np.newaxis],
        port_mapping=[(0, 1), (1, 0), (2, 2)],
    )

    on_axis = scene_projector.compute_point(0, 0)

    def null_loss(device):
        out = device.calculate_outputs(on_axis)
        return out[0, 0] / (out[0, 0] + out[0, 1] + 1e-12)

    stage = OptimizationStage(
        name='nulling',
        tunable_parameters=['ps1', 'ps2'],
        loss_fn=null_loss,
        bounds={'ps1': (-np.pi, np.pi)},
    )

    result = Experiment('run1', chip, lantern, [stage]).run(verbose=True)
    result.summary()
    result.save('results/')
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from PLsim.system.device import Device
from PLsim.system.pic import ActivePIC
from PLsim.utils.diagnostics import plot_maps


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ChipDesign:
    """Describes a PIC chip architecture.

    Parameters
    ----------
    name:
        Human-readable identifier for this design.
    device_template:
        Callable that accepts keyword parameter values and returns a PIC device.
        This is the same function you would pass to ``ActivePIC``.
    parameter_names:
        Ordered list of all tunable parameter names accepted by ``device_template``.
    initial_parameters:
        Default starting values for every parameter in ``parameter_names``.
    port_labels:
        Optional mapping from internal port names to human-readable labels used
        in plots and summary output.  E.g. ``{'a': 'null a', 'b': 'antinull a'}``.
    """

    name: str
    device_template: Callable
    parameter_names: list[str]
    initial_parameters: dict[str, float]
    port_labels: dict[str, str] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        """Save chip design to a JSON file (excludes ``device_template``).

        The file is written to ``<path>/<name>.json``.

        Parameters
        ----------
        path:
            Directory where the JSON file will be saved.
        """
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "name": self.name,
            "parameter_names": self.parameter_names,
            "initial_parameters": self.initial_parameters,
            "port_labels": self.port_labels,
        }
        out_path = out_dir / f"{self.name}.json"
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved → {out_path}")

    @classmethod
    def load(cls, path: str | Path, device_template: Callable) -> ChipDesign:
        """Load a chip design from a JSON file.

        The ``device_template`` callable must be provided at load time because
        Python functions cannot be serialized to JSON.

        Parameters
        ----------
        path:
            Path to the ``.json`` file (or a directory containing
            ``<name>.json``).
        device_template:
            The same callable used when the design was originally created.
        """
        p = Path(path)
        with open(p) as f:
            data = json.load(f)
        return cls(
            name=data["name"],
            device_template=device_template,
            parameter_names=data["parameter_names"],
            initial_parameters=data["initial_parameters"],
            port_labels=data.get("port_labels", {}),
        )


@dataclass
class LanternConfig:
    """Describes the photonic lantern connecting to the chip.

    Parameters
    ----------
    name:
        Human-readable identifier, e.g. ``'MSPL'`` or ``'SPL_eps03'``.
    matrix:
        Complex transfer matrix with shape ``(n_wavelengths, n_modes, n_modes)``.
    port_mapping:
        List of ``(lantern_port_idx, pic_port_idx)`` pairs passed to ``Device``.
    """

    name: str
    matrix: np.ndarray
    port_mapping: list[tuple[int, int]]


@dataclass
class OptimizationStage:
    """One stage of a sequential optimization.

    Only the parameters listed in ``tunable_parameters`` are updated by the
    optimizer.  All other parameters keep whatever values were set by the
    previous stage (or ``ChipDesign.initial_parameters`` for the first stage).

    Parameters
    ----------
    name:
        Label for this stage, used in summaries and saved results.
    tunable_parameters:
        Subset of the chip's ``parameter_names`` to optimize in this stage.
    loss_fn:
        User-defined callable ``(device: Device) -> float``.  Called on every
        optimizer iteration.  Capture projections and other data via closure.
    bounds:
        Optional per-parameter ``(low, high)`` bounds.  Parameters not listed
        here are unbounded.
    initial_guess:
        Optional override for the starting values of ``tunable_parameters``.
        If ``None``, values are taken from the parameters carried over from the
        previous stage.
    method:
        scipy optimizer method, e.g. ``'Powell'`` (default) or ``'L-BFGS-B'``.
    options:
        Extra keyword arguments forwarded to ``scipy.optimize.minimize`` as the
        ``options`` dict (e.g. ``{'maxiter': 500}``).
    """

    name: str
    tunable_parameters: list[str]
    loss_fn: Callable[[Device], float]
    bounds: dict[str, tuple[float, float]] | None = None
    initial_guess: dict[str, float] | None = None
    method: str = 'Powell'
    options: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    """Outcome of a single optimization stage.

    Attributes
    ----------
    name:
        Matches ``OptimizationStage.name``.
    success:
        Whether the optimizer reported success.
    optimized_params:
        Parameter values at the end of this stage (tunable parameters only).
    loss_value:
        Loss function value at the optimized parameters.
    n_evals:
        Number of loss function evaluations performed.
    raw_result:
        The raw ``scipy.optimize.OptimizeResult`` object.
    """

    name: str
    success: bool
    optimized_params: dict[str, float]
    loss_value: float
    n_evals: int
    raw_result: Any


@dataclass
class ExperimentResult:
    """Full result of an ``Experiment.run()`` call.

    Attributes
    ----------
    experiment_name, chip_name, lantern_name:
        Identifiers from the originating ``Experiment`` and its components.
    stage_results:
        One ``StageResult`` per ``OptimizationStage``, in order.
    final_params:
        Complete parameter dict after all stages have run.
    """

    experiment_name: str
    chip_name: str
    lantern_name: str
    stage_results: list[StageResult]
    final_params: dict[str, float]

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print a compact table of stage outcomes to stdout."""
        header = f"{'Stage':<20} {'Success':<10} {'Loss':>12} {'Evals':>8}"
        print(f"\nExperiment: {self.experiment_name}  |  chip: {self.chip_name}  |  lantern: {self.lantern_name}")
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        for sr in self.stage_results:
            ok = "yes" if sr.success else "NO"
            print(f"{sr.name:<20} {ok:<10} {sr.loss_value:>12.6g} {sr.n_evals:>8}")
        print("-" * len(header))

    def plot_losses(self, ax=None, **bar_kwargs) -> None:
        """Bar chart of the final loss value per stage."""
        names = [sr.name for sr in self.stage_results]
        losses = [sr.loss_value for sr in self.stage_results]

        if ax is None:
            _, ax = plt.subplots(figsize=(max(4, len(names) * 1.2), 4))

        ax.bar(names, losses, **bar_kwargs)
        ax.set_ylabel("Loss")
        ax.set_title(f"{self.experiment_name}  ({self.chip_name} / {self.lantern_name})")
        ax.tick_params(axis='x', rotation=30)
        plt.tight_layout()

    def save(self, path: str | Path) -> None:
        """Save result to a JSON file.

        The file is written to ``<path>/<experiment_name>.json``, creating
        ``path`` if it does not exist.

        Parameters
        ----------
        path:
            Directory where the JSON file will be saved.
        """
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "experiment_name": self.experiment_name,
            "chip_name": self.chip_name,
            "lantern_name": self.lantern_name,
            "final_params": self.final_params,
            "stages": [
                {
                    "name": sr.name,
                    "success": sr.success,
                    "loss_value": sr.loss_value,
                    "n_evals": sr.n_evals,
                    "optimized_params": sr.optimized_params,
                }
                for sr in self.stage_results
            ],
        }

        out_path = out_dir / f"{self.experiment_name}.json"
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved → {out_path}")

    @classmethod
    def load(cls, path: str | Path) -> ExperimentResult:
        """Load a previously saved result from a JSON file.

        Parameters
        ----------
        path:
            Path to the ``.json`` file (or a directory containing
            ``<experiment_name>.json``).
        """
        p = Path(path)
        with open(p) as f:
            data = json.load(f)

        stage_results = [
            StageResult(
                name=s["name"],
                success=s["success"],
                optimized_params=s["optimized_params"],
                loss_value=s["loss_value"],
                n_evals=s["n_evals"],
                raw_result=None,
            )
            for s in data["stages"]
        ]

        return cls(
            experiment_name=data["experiment_name"],
            chip_name=data["chip_name"],
            lantern_name=data["lantern_name"],
            stage_results=stage_results,
            final_params=data["final_params"],
        )


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

class Experiment:
    """Orchestrates a multi-stage chip optimization experiment.

    Parameters
    ----------
    name:
        Identifier for this run (used in results and saved filenames).
    chip:
        The ``ChipDesign`` to optimize.
    lantern:
        The ``LanternConfig`` to pair with the chip.
    stages:
        Ordered list of ``OptimizationStage`` objects.  Each stage starts from
        the parameter values left by the previous stage.
    """

    def __init__(
        self,
        name: str,
        chip: ChipDesign,
        lantern: LanternConfig,
        stages: list[OptimizationStage],
    ) -> None:
        self.name = name
        self.chip = chip
        self.lantern = lantern
        self.stages = stages

    def build_device(self, params: dict[str, float] | None = None) -> Device:
        """Construct an ``ActivePIC``-backed ``Device`` from chip + lantern.

        Parameters
        ----------
        params:
            Parameter values to initialise the device with.  Defaults to
            ``chip.initial_parameters``.
        """
        if params is None:
            params = self.chip.initial_parameters

        active_pic = ActivePIC(
            device_template=self.chip.device_template,
            parameter_names=self.chip.parameter_names,
            initial_parameters=params,
            name=self.chip.name,
        )

        return Device(
            self.lantern.matrix,
            pic_device=active_pic,
            port_mapping=self.lantern.port_mapping,
            verbose=False,
        )

    def run(self, verbose: bool = False) -> ExperimentResult:
        """Run all optimization stages sequentially and return the results.

        Parameters
        ----------
        verbose:
            If ``True``, print stage progress to stdout.
        """
        current_params = dict(self.chip.initial_parameters)
        device = self.build_device(current_params)

        stage_results: list[StageResult] = []

        for stage in self.stages:
            if verbose:
                print(f"[{self.name}] Stage '{stage.name}' — tuning: {stage.tunable_parameters}")

            # Initial values for this stage's tunable parameters
            x0_dict = stage.initial_guess or {k: current_params[k] for k in stage.tunable_parameters}
            x0 = np.array([x0_dict[k] for k in stage.tunable_parameters], dtype=float)

            # Bounds (scipy wants a list aligned to x0)
            if stage.bounds:
                bounds_list = [stage.bounds.get(k, (None, None)) for k in stage.tunable_parameters]
            else:
                bounds_list = None

            # Wrap the user's loss function so scipy can call it
            def _objective(params_array, _stage=stage, _device=device):
                kwargs = dict(zip(_stage.tunable_parameters, params_array))
                _device.update_pic_matrix(**kwargs)
                return _stage.loss_fn(_device)

            opt = minimize(
                _objective,
                x0,
                method=stage.method,
                bounds=bounds_list,
                options=stage.options or None,
            )

            optimized = {k: float(v) for k, v in zip(stage.tunable_parameters, opt.x)}

            # Carry optimized values into current_params for the next stage
            current_params.update(optimized)

            # Ensure the device reflects the final optimized state
            device.update_pic_matrix(**optimized)

            sr = StageResult(
                name=stage.name,
                success=bool(opt.success),
                optimized_params=optimized,
                loss_value=float(opt.fun),
                n_evals=int(opt.nfev),
                raw_result=opt,
            )
            stage_results.append(sr)

            if verbose:
                status = "OK" if opt.success else "FAILED"
                print(f"  [{status}] loss={opt.fun:.6g}  evals={opt.nfev}")

        return ExperimentResult(
            experiment_name=self.name,
            chip_name=self.chip.name,
            lantern_name=self.lantern.name,
            stage_results=stage_results,
            final_params=current_params,
        )

    def plot_output_maps(
        self,
        out_grid: np.ndarray,
        fov: float,
        params: dict[str, float] | None = None,
        labels: list[str] | None = None,
        return_results: bool = False,
        **plot_kwargs,
    ) -> np.ndarray | None:
        """Build a device and plot per-port 2-D output intensity maps.

        Parameters
        ----------
        out_grid:
            Projection grid from ``SceneProjector.compute_point_grid``.
        fov:
            Full field of view in radians.
        params:
            Parameter values to use.  Defaults to
            ``chip.initial_parameters``.  Pass ``result.final_params``
            to visualise the post-optimisation device state.
        labels:
            Port labels.  Defaults to ``chip.port_labels`` values (in
            port order) or raw port names.
        return_results:
            If ``True``, return the raw output array.
        **plot_kwargs:
            Forwarded to :func:`plot_maps` (e.g. ``ncols``, ``figsize``).
        """
        if params is None:
            params = self.chip.initial_parameters

        device = self.build_device(params)

        if labels is None and self.chip.port_labels:
            port_order = device.pic.output_names
            labels = [self.chip.port_labels.get(p, p) for p in port_order]

        return plot_maps(device, out_grid, fov, labels=labels,
                         return_results=return_results, **plot_kwargs)


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------

def compare_experiments(
    results: list[ExperimentResult],
    stage: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> None:
    """Grouped bar chart comparing loss values across experiments.

    Parameters
    ----------
    results:
        List of ``ExperimentResult`` objects to compare.
    stage:
        If given, only plot the loss for that stage name.  Otherwise all stages
        are shown as grouped bars.
    figsize:
        Override the default figure size.
    """
    if not results:
        return

    # Collect stage names (union across all results, preserving first-seen order)
    all_stages: list[str] = []
    for r in results:
        for sr in r.stage_results:
            if sr.name not in all_stages:
                all_stages.append(sr.name)

    if stage is not None:
        if stage not in all_stages:
            raise ValueError(f"Stage '{stage}' not found in results. Available: {all_stages}")
        all_stages = [stage]

    exp_names = [r.experiment_name for r in results]
    n_exp = len(results)
    n_stages = len(all_stages)

    # Build loss matrix: shape (n_stages, n_exp)
    loss_matrix = np.full((n_stages, n_exp), np.nan)
    for j, r in enumerate(results):
        stage_map = {sr.name: sr.loss_value for sr in r.stage_results}
        for i, sname in enumerate(all_stages):
            if sname in stage_map:
                loss_matrix[i, j] = stage_map[sname]

    fig, ax = plt.subplots(figsize=figsize or (max(6, n_exp * n_stages * 0.8), 4))

    width = 0.8 / n_stages
    offsets = np.linspace(-(n_stages - 1) / 2, (n_stages - 1) / 2, n_stages) * width
    x = np.arange(n_exp)

    for i, (sname, offset) in enumerate(zip(all_stages, offsets)):
        ax.bar(x + offset, loss_matrix[i], width=width, label=sname)

    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, rotation=30, ha='right')
    ax.set_ylabel("Loss")
    ax.set_title("Experiment comparison")
    ax.legend(title="Stage")
    plt.tight_layout()
    plt.show()
