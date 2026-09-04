"""Nonlinear least-squares calibration of the degradation coefficients.

Fits the model in :mod:`aacopt.degradation` to measured fractional capacity
loss from NASA RW reference discharges. Contains no optimization-result
dependency of any kind (RULE 6): the only inputs are the stage-1 measurement
dataset and ``configs/degradation.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc

from aacopt.degradation import (
    DegradationParameters,
    HybridDegradationModel,
    calendar_from_anchors,
)

PARAM_ORDER_CAL = ["log10_A_cal", "B_cal", "C_cal", "Ea_cal", "z_cal"]


@dataclass
class CellData:
    """Stress history and measured loss for one cell, in interval order."""

    cell: str
    duration_h: np.ndarray
    mean_soc: np.ndarray
    mean_temperature_c: np.ndarray          # duration-weighted (calendar)
    cyclic_mean_temperature_c: np.ndarray   # throughput-weighted (cyclic)
    dah: np.ndarray             # (n_intervals, n_bins)
    y: np.ndarray               # measured fractional capacity loss
    cum_ah: np.ndarray
    cum_duration_h: np.ndarray
    t_end_h: np.ndarray

    @property
    def n(self) -> int:
        return int(self.y.size)


@dataclass(frozen=True)
class ParameterLayout:
    """Maps a flat optimization vector to model coefficients.

    The reductions below are not cosmetic — they are forced by the information
    content of the NASA RW record (quantified in
    ``results/04_degradation_validation/identifiability.json``):

    ``fit_C_cal``
        ``exp(B_cal*SOC) * exp(-C_cal*SOC/(R*T))`` collapses to
        ``exp(SOC*(B_cal - C_cal/(R*T)))``, so ``B_cal`` and ``C_cal`` are
        *structurally aliased* at fixed temperature and separate only through
        the temperature dependence of the SOC term. With a 19-39 degC span the
        pair is unidentifiable (empirically |corr| = 1.0000). Default: hold
        ``C_cal = 0`` and let ``B_cal`` carry the SOC sensitivity, rather than
        report an arbitrary bound-active value for both.

    ``shared_ea_cyc`` / ``shared_z_cyc``
        The three C-rate bins' cumulative throughputs are collinear
        (r = 0.87-0.98), which cannot support three independent
        ``(B, Ea, z)`` triples. Default: one activation energy and one exponent
        shared across bins, with the C-rate dependence carried by ``B_cyc(C)``
        — which is what the model's C-rate grid exists for.

    Reference-condition centring
    ----------------------------
    The prefactors are *not* fitted directly. An Arrhenius prefactor and its
    activation energy are near-perfectly correlated when fitted raw
    (empirically |corr| = 0.9923 for ``A_cal`` <-> ``Ea_cal``), because
    ``A*exp(-Ea/(R*T))`` is unchanged by trading one against the other over a
    narrow temperature span; the optimizer escapes the resulting ridge by
    pinning ``A_cal`` at a bound. So the fitted quantities are the rate
    coefficients *at a reference condition* inside the data,

        k_cal_ref = A_cal * exp(B_cal*SOC_ref) * exp(-Ea_cal/(R*T_ref))
        k_cyc_ref(C) = B_cyc(C) * exp(-Ea_cyc(C)/(R*T_ref))

    with ``A_cal`` and ``B_cyc`` recovered analytically afterwards. This is a
    change of variables only — the model equation, its predictions and its
    degrees of freedom are identical. It just puts the estimated parameter
    where the data is instead of at a physically unreachable ``T -> inf``
    intercept.
    """

    c_rate_bins: Tuple[float, ...]
    fit_C_cal: bool = False
    shared_ea_cyc: bool = True
    shared_z_cyc: bool = True
    fixed_C_cal: float = 0.0
    monotone_B_cyc: bool = True
    soc_ref: float = 0.5
    t_ref_k: float = 298.15
    fit_calendar: bool = False
    calendar: Optional[Tuple[float, float, float, float, float]] = None
    c_rate_power_law: bool = True
    c_rate_ref: float = 1.0
    scale_cells: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.scale_cells and not self.c_rate_power_law:
            raise ValueError("per-cell scaling requires the C-rate power law")
        if not self.fit_calendar and self.calendar is None:
            raise ValueError(
                "fit_calendar=False requires the fixed calendar coefficients "
                "(A_cal, B_cal, C_cal, Ea_cal, z_cal); derive them with "
                "aacopt.degradation.calendar_from_anchors"
            )

    @property
    def names(self) -> List[str]:
        n = len(self.c_rate_bins)
        names: List[str] = []
        if self.fit_calendar:
            names += ["log10_k_cal_ref", "B_cal"]
            if self.fit_C_cal:
                names.append("C_cal")
            names += ["Ea_cal", "z_cal"]
        if self.scale_cells:
            # one cell-specific scale each, then the SHARED shape parameters
            names += [
                f"log10_k_cyc_ref[{self.c_rate_ref:g}C|{c}]" for c in self.scale_cells
            ]
            names.append("p_c_rate")
        elif self.c_rate_power_law:
            names += [f"log10_k_cyc_ref[{self.c_rate_ref:g}C]", "p_c_rate"]
        elif self.monotone_B_cyc:
            names.append(f"log10_k_cyc_ref_base[{self.c_rate_bins[0]:g}C]")
            names += [
                f"dlog10_k_cyc_ref[{self.c_rate_bins[b]:g}C]" for b in range(1, n)
            ]
        else:
            names += [f"log10_k_cyc_ref[{self.c_rate_bins[b]:g}C]" for b in range(n)]
        if self.shared_ea_cyc:
            names.append("Ea_cyc[shared]")
        else:
            names += [f"Ea_cyc[{self.c_rate_bins[b]:g}C]" for b in range(n)]
        if self.shared_z_cyc:
            names.append("z_cyc[shared]")
        else:
            names += [f"z_cyc[{self.c_rate_bins[b]:g}C]" for b in range(n)]
        return names

    @property
    def size(self) -> int:
        return len(self.names)

    @property
    def label(self) -> str:
        bits = [f"{self.size}p"]
        if self.fit_calendar:
            bits.append("calendar FITTED")
            bits.append("C_cal free" if self.fit_C_cal else "C_cal=0")
        else:
            bits.append("calendar literature-anchored")
        bits.append("Ea_cyc shared" if self.shared_ea_cyc else "Ea_cyc per-bin")
        bits.append("z_cyc shared" if self.shared_z_cyc else "z_cyc per-bin")
        if self.scale_cells:
            bits.append(
                f"k_cyc ~ C^p about {self.c_rate_ref:g}C, "
                f"per-cell scale ({len(self.scale_cells)} cells), shape shared"
            )
        elif self.c_rate_power_law:
            bits.append(f"k_cyc ~ C^p about {self.c_rate_ref:g}C")
        else:
            bits.append("k_cyc monotone grid" if self.monotone_B_cyc else "k_cyc free grid")
        return ", ".join(bits)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_parameters": self.size,
            "parameter_names": self.names,
            "label": self.label,
            "fit_calendar": self.fit_calendar,
            "fixed_calendar_coefficients": (
                None if self.calendar is None else {
                    k: float(v) for k, v in zip(
                        ("A_cal", "B_cal", "C_cal", "Ea_cal", "z_cal"), self.calendar,
                    )
                }
            ),
            "fit_C_cal": self.fit_C_cal,
            "fixed_C_cal": self.fixed_C_cal,
            "shared_ea_cyc": self.shared_ea_cyc,
            "shared_z_cyc": self.shared_z_cyc,
            "monotone_B_cyc": self.monotone_B_cyc,
            "c_rate_power_law": self.c_rate_power_law,
            "c_rate_ref": self.c_rate_ref,
            "scale_cells": list(self.scale_cells),
            "shared_shape_parameters": (
                ["p_c_rate", "Ea_cyc", "z_cyc"] if self.scale_cells else None
            ),
            "reference_condition": {"soc_ref": self.soc_ref, "T_ref_K": self.t_ref_k},
            "parameterization": (
                "cyclic prefactors fitted as rate coefficients at T_ref; "
                "B_cyc recovered analytically"
            ),
        }

    def to_params(
        self,
        x: np.ndarray,
        *,
        q_nominal_ah: float,
        R: float,
        provenance: str,
        cell: Optional[str] = None,
    ) -> DegradationParameters:
        """Coefficients for one cell.

        With ``scale_cells`` set, the cyclic prefactor is cell-specific and
        every shape parameter (``p_c_rate``, ``Ea_cyc``, ``z_cyc``) is shared,
        so ``cell`` selects which scale to apply. Measured per-ampere-hour
        damage differs 1.6-1.8x between these four cells at matched throughput
        — see the ``cell_variability`` block of ``configs/degradation.yaml``.
        """
        n = len(self.c_rate_bins)
        i = 0
        if self.fit_calendar:
            log_k_cal = x[i]; i += 1
            b_cal = x[i]; i += 1
            if self.fit_C_cal:
                c_cal = x[i]; i += 1
            else:
                c_cal = self.fixed_C_cal
            ea_cal = x[i]; i += 1
            z_cal = x[i]; i += 1
            rt_ref = R * self.t_ref_k
            a_cal = 10.0 ** (
                log_k_cal
                + (-b_cal * self.soc_ref + (ea_cal + c_cal * self.soc_ref) / rt_ref)
                / np.log(10.0)
            )
        else:
            a_cal, b_cal, c_cal, ea_cal, z_cal = self.calendar
        if self.scale_cells:
            if cell is None:
                raise ValueError("per-cell scaling requires the cell name")
            try:
                j = self.scale_cells.index(cell)
            except ValueError:
                raise ValueError(
                    f"no calibrated scale for cell {cell!r}; "
                    f"have {list(self.scale_cells)}"
                ) from None
            log_k_ref = x[i + j]
            i += len(self.scale_cells)
            p_c = max(float(x[i]), 0.0); i += 1
            log_k_cyc = log_k_ref + p_c * np.log10(
                np.asarray(self.c_rate_bins, dtype=np.float64) / self.c_rate_ref
            )
        elif self.c_rate_power_law:
            # k_cyc_ref(C) = k_ref * (C / C_ref) ** p,  p >= 0
            log_k_ref = x[i]; i += 1
            p_c = max(float(x[i]), 0.0); i += 1
            log_k_cyc = log_k_ref + p_c * np.log10(
                np.asarray(self.c_rate_bins, dtype=np.float64) / self.c_rate_ref
            )
        elif self.monotone_B_cyc:
            # base + non-negative increments => k_cyc non-decreasing in C-rate
            log_k_cyc = np.cumsum(
                np.concatenate([[x[i]], np.maximum(x[i + 1:i + n], 0.0)])
            )
            i += n
        else:
            log_k_cyc = x[i:i + n]; i += n
        if self.shared_ea_cyc:
            ea_cyc = np.full(n, x[i]); i += 1
        else:
            ea_cyc = x[i:i + n]; i += n
        if self.shared_z_cyc:
            z_cyc = np.full(n, x[i]); i += 1
        else:
            z_cyc = x[i:i + n]; i += n

        # undo the reference-condition centring to recover the coefficient the
        # model equation is written in terms of
        log_b = log_k_cyc + ea_cyc / (R * self.t_ref_k * np.log(10.0))
        return DegradationParameters(
            A_cal=float(a_cal),
            B_cal=float(b_cal),
            C_cal=float(c_cal),
            Ea_cal=float(ea_cal),
            z_cal=float(z_cal),
            c_rate_bins=tuple(float(c) for c in self.c_rate_bins),
            B_cyc=tuple(float(10.0 ** v) for v in log_b),
            Ea_cyc=tuple(float(v) for v in ea_cyc),
            z_cyc=tuple(float(v) for v in z_cyc),
            R=float(R),
            q_nominal_ah=float(q_nominal_ah),
            provenance=provenance,
        )

    def bounds(self, cfg_bounds: Dict[str, Sequence[float]]) -> Tuple[np.ndarray, np.ndarray]:
        n = len(self.c_rate_bins)
        lo: List[float] = []
        hi: List[float] = []

        def add(key: str, times: int = 1) -> None:
            b = cfg_bounds[key]
            for _ in range(times):
                lo.append(float(b[0]))
                hi.append(float(b[1]))

        if self.fit_calendar:
            add("log10_k_cal_ref")
            add("B_cal")
            if self.fit_C_cal:
                add("C_cal")
            add("Ea_cal")
            add("z_cal")
        if self.scale_cells:
            add("log10_k_cyc_ref", len(self.scale_cells))
            add("p_c_rate")
        elif self.c_rate_power_law:
            add("log10_k_cyc_ref")
            add("p_c_rate")
        elif self.monotone_B_cyc:
            add("log10_k_cyc_ref_base")
            add("dlog10_k_cyc_ref", n - 1)
        else:
            add("log10_k_cyc_ref", n)
        add("Ea_cyc", 1 if self.shared_ea_cyc else n)
        add("z_cyc", 1 if self.shared_z_cyc else n)
        return np.asarray(lo), np.asarray(hi)


def layouts_from_config(
    cfg: Dict[str, Any],
    *,
    R: float,
    calendar_scale: float = 1.0,
    cells: Sequence[str] = (),
) -> Dict[str, ParameterLayout]:
    """Named parameterizations: the deployed one plus documented ablations.

    The calendar branch is literature-anchored (derived from
    ``cfg["calendar_branch"]["anchors"]``) unless
    ``cfg["calibration"]["fit_calendar_branch"]`` is true, in which case the
    ``fit_calendar`` ablations below expose what fitting it does to
    identifiability.
    """
    b = tuple(float(c) for c in cfg["c_rate_bins"])
    cal_cfg = cfg["calibration"]
    ref = cal_cfg["reference_condition"]
    cal = calendar_from_anchors(
        cfg["calendar_branch"]["anchors"], R=R, scale=calendar_scale,
    )
    fixed = (cal["A_cal"], cal["B_cal"], cal["C_cal"], cal["Ea_cal"], cal["z_cal"])

    scale_cells = tuple(str(c).upper() for c in cells)

    def mk(
        sh_ea: bool, sh_z: bool, mono: bool,
        *, fit_cal: bool = False, fit_c: bool = False, power_law: bool = True,
        per_cell: bool = False,
    ) -> ParameterLayout:
        return ParameterLayout(
            b, fit_c, sh_ea, sh_z,
            monotone_B_cyc=mono,
            soc_ref=float(ref["soc_ref"]), t_ref_k=float(ref["T_ref_K"]),
            fit_calendar=fit_cal, calendar=fixed,
            c_rate_power_law=power_law,
            c_rate_ref=float(cal_cfg["c_rate_reference"]),
            scale_cells=scale_cells if (per_cell and power_law) else (),
        )

    return {
        # deployed: one coefficient set for all four cells. Protocol ranking
        # depends on the SHARED C-rate exponent, which this identifies.
        # Per-cell scales improve in-sample RMSE but pin p_c_rate at a bound
        # and are not used downstream.
        "primary": mk(True, True, True, per_cell=False),
        "per_cell_scale": mk(True, True, True, per_cell=True),
        # ablations relaxing each reduction in turn
        "grid_monotone": mk(True, True, True, power_law=False),
        "grid_free": mk(True, True, False, power_law=False),
        "free_z_cyc": mk(True, False, True),
        "free_ea_cyc": mk(False, True, True),
        # ablations that ALSO fit the calendar branch, retained as the
        # documented evidence for why it is anchored instead (RULE 8)
        "fit_calendar": mk(True, True, True, fit_cal=True),
        "fit_calendar_full": mk(
            False, False, False, fit_cal=True, fit_c=True, power_law=False,
            per_cell=False,
        ),
    }


def predict_cell(params: DegradationParameters, data: CellData) -> np.ndarray:
    return HybridDegradationModel(params).accumulate_intervals(
        duration_h=data.duration_h,
        mean_soc=data.mean_soc,
        mean_temperature_c=data.mean_temperature_c,
        dah=data.dah,
        temperature_c_cyclic=data.cyclic_mean_temperature_c,
    )


def predict_cell_split(params: DegradationParameters, data: CellData) -> Dict[str, np.ndarray]:
    return HybridDegradationModel(params).cumulative_split(
        duration_h=data.duration_h,
        mean_soc=data.mean_soc,
        mean_temperature_c=data.mean_temperature_c,
        dah=data.dah,
        temperature_c_cyclic=data.cyclic_mean_temperature_c,
    )


def residuals(
    x: np.ndarray,
    layout: ParameterLayout,
    datasets: Sequence[CellData],
    *,
    q_nominal_ah: float,
    R: float,
) -> np.ndarray:
    parts = []
    for d in datasets:
        params = layout.to_params(
            x, q_nominal_ah=q_nominal_ah, R=R, provenance="trial", cell=d.cell,
        )
        try:
            yhat = predict_cell(params, d)
        except (FloatingPointError, OverflowError, ValueError):
            yhat = np.full(d.n, 1e3)
        r = np.clip(yhat, -1e3, 1e3) - d.y
        parts.append(np.where(np.isfinite(r), r, 1e3))
    return np.concatenate(parts)


def _anchor_start(
    layout: ParameterLayout,
    anchor: Dict[str, float],
    datasets: Sequence[CellData],
    lo: np.ndarray,
    hi: np.ndarray,
) -> np.ndarray:
    """Literature-anchored start: exponents and activation energies from the
    cited source, reference-condition rate coefficients set so end-of-record
    loss is order 0.5 (half calendar, half cyclic).
    """
    n = len(layout.c_rate_bins)
    ea_cyc = float(anchor["Ea_cyc"])
    z_cyc = float(anchor["z_cyc"])

    d = datasets[0]
    ah_end = float(np.sum(d.dah))

    # the fitted quantities are rates AT the reference condition, so the start
    # follows directly from the target loss without any Arrhenius factor
    if layout.fit_calendar:
        z_cal = float(anchor["z_cal"])
        t_end = float(np.sum(d.duration_h))
        # split the target loss half calendar / half cyclic at the start point
        log_k_cal = np.log10(max(0.25 / max(t_end ** z_cal, 1e-12), 1e-30))
        target_cyc = 0.25
    else:
        # calendar is fixed, so the cyclic branch starts by carrying the whole
        # residual loss the anchored calendar term does not already explain
        z_cal = layout.calendar[4]
        target_cyc = 0.5
    log_k_cyc = np.log10(max(target_cyc / max((ah_end / n) ** z_cyc, 1e-12), 1e-30))

    x: List[float] = []
    if layout.fit_calendar:
        x += [log_k_cal, float(anchor["B_cal"])]
        if layout.fit_C_cal:
            x.append(float(anchor["C_cal"]))
        x += [float(anchor["Ea_cal"]), z_cal]
    if layout.scale_cells:
        x += [log_k_cyc] * len(layout.scale_cells)
        x += [float(anchor.get("p_c_rate", 1.0))]
    elif layout.c_rate_power_law:
        x += [log_k_cyc, float(anchor.get("p_c_rate", 1.0))]
    elif layout.monotone_B_cyc:
        x += [log_k_cyc] + [0.0] * (n - 1)
    else:
        x += [log_k_cyc] * n
    x += [ea_cyc] if layout.shared_ea_cyc else [ea_cyc] * n
    x += [z_cyc] if layout.shared_z_cyc else [z_cyc] * n
    return np.clip(np.asarray(x, dtype=np.float64), lo, hi)


class CellParameters:
    """Coefficient sets for a group of cells: shared shape, per-cell scale.

    Measured per-ampere-hour damage differs 1.6-1.8x across RW9-RW12 at
    matched throughput and this is not explained by any stress descriptor
    available in the dataset (RW11 and RW12 sit at 24.5 vs 24.7 degC and need
    672 vs 1108 Ah to reach 20 % loss). A single pooled coefficient set
    therefore cannot represent all four cells, so the *shape* of the damage
    law is shared and one scale factor per cell absorbs the spread.

    Behaves like a mapping ``cell -> DegradationParameters``.
    """

    def __init__(
        self,
        layout: "ParameterLayout",
        x: np.ndarray,
        *,
        q_nominal_ah: float,
        R: float,
        provenance: str,
    ) -> None:
        self.layout = layout
        self.x = np.asarray(x, dtype=np.float64)
        self.cells = tuple(layout.scale_cells)
        self._by_cell = {
            c: layout.to_params(
                self.x, q_nominal_ah=q_nominal_ah, R=R,
                provenance=f"{provenance}|{c}", cell=c,
            )
            for c in self.cells
        }

    def __getitem__(self, cell: str) -> DegradationParameters:
        try:
            return self._by_cell[cell]
        except KeyError:
            raise KeyError(
                f"no calibrated coefficients for cell {cell!r}; "
                f"have {list(self.cells)}"
            ) from None

    def __contains__(self, cell: object) -> bool:
        return cell in self._by_cell

    def __iter__(self):
        return iter(self._by_cell)

    def items(self):
        return self._by_cell.items()

    @property
    def representative(self) -> DegradationParameters:
        """First cell's set — for reporting the SHARED shape only.

        Never use this to score a specific cell; its scale belongs to one cell.
        """
        return self._by_cell[self.cells[0]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "per_cell": {c: p.to_dict() for c, p in self._by_cell.items()},
            "shared_shape": {
                "p_c_rate": (
                    float(self.x[self.layout.names.index("p_c_rate")])
                    if "p_c_rate" in self.layout.names else None
                ),
                "Ea_cyc": float(self.representative.Ea_cyc[0]),
                "z_cyc": float(self.representative.z_cyc[0]),
                "A_cal": float(self.representative.A_cal),
                "B_cal": float(self.representative.B_cal),
                "C_cal": float(self.representative.C_cal),
                "Ea_cal": float(self.representative.Ea_cal),
                "z_cal": float(self.representative.z_cal),
            },
            "cell_scale_log10_k_cyc_ref": {
                c: float(np.log10(p.B_cyc[
                    int(np.argmin(np.abs(
                        np.asarray(p.c_rate_bins) - self.layout.c_rate_ref
                    )))
                ]) - p.Ea_cyc[0] / (p.R * self.layout.t_ref_k * np.log(10.0)))
                for c, p in self._by_cell.items()
            },
        }


ParamsLike = Any  # DegradationParameters or CellParameters


def params_for(params: ParamsLike, cell: str) -> DegradationParameters:
    """Resolve the coefficient set that applies to ``cell``."""
    if isinstance(params, CellParameters):
        return params[cell]
    return params


@dataclass
class FitResult:
    params: DegradationParameters
    layout: ParameterLayout
    x: np.ndarray
    cost: float
    n_obs: int
    n_params: int
    success: bool
    status: int
    message: str
    n_starts: int
    start_costs: List[float]
    standard_errors: Optional[np.ndarray]
    correlation: Optional[np.ndarray]
    jtj_condition: Optional[float]
    flags: List[str] = field(default_factory=list)

    def metrics(self, datasets: Sequence[CellData]) -> Dict[str, Any]:
        return goodness_of_fit(self.params, datasets)

    def to_dict(self, datasets: Optional[Sequence[CellData]] = None) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "parameters": self.params.to_dict(),
            "per_cell_scaled": isinstance(self.params, CellParameters),
            "layout": self.layout.to_dict(),
            "parameter_names": self.layout.names,
            "x": [float(v) for v in self.x],
            "cost": self.cost,
            "rss": 2.0 * self.cost,
            "n_obs": self.n_obs,
            "n_params": self.n_params,
            "dof": self.n_obs - self.n_params,
            "success": self.success,
            "status": self.status,
            "message": self.message,
            "n_starts": self.n_starts,
            "start_cost_min": min(self.start_costs) if self.start_costs else None,
            "start_cost_median": float(np.median(self.start_costs)) if self.start_costs else None,
            "start_cost_max": max(self.start_costs) if self.start_costs else None,
            "n_starts_within_1pct_of_best": int(
                sum(1 for c in self.start_costs if c <= 1.01 * min(self.start_costs))
            ) if self.start_costs else None,
            "standard_errors": None if self.standard_errors is None
            else [float(v) for v in self.standard_errors],
            "correlation_matrix": None if self.correlation is None
            else [[float(v) for v in row] for row in self.correlation],
            "jtj_condition_number": self.jtj_condition,
            "identifiability_flags": self.flags,
        }
        if datasets is not None:
            d["goodness_of_fit"] = self.metrics(datasets)
            d["calendar_share_end_of_record"] = calendar_share(self.params, datasets)
        return d


def goodness_of_fit(
    params: DegradationParameters,
    datasets: Sequence[CellData],
    *,
    window_max_y: Optional[float] = None,
) -> Dict[str, Any]:
    """Pooled and per-cell metrics, optionally also inside an operating window.

    ``window_max_y`` restricts the metric (never the fit) to references whose
    measured loss is below the threshold — the regime in which the downstream
    optimization and retention projections are read.
    """
    per_cell: Dict[str, Any] = {}
    all_y: List[np.ndarray] = []
    all_p: List[np.ndarray] = []
    for d in datasets:
        yhat = predict_cell(params_for(params, d.cell), d)
        per_cell[d.cell] = _metrics(d.y, yhat)
        if window_max_y is not None:
            m = d.y <= float(window_max_y)
            per_cell[d.cell]["window"] = (
                _metrics(d.y[m], yhat[m]) if m.sum() >= 3 else None
            )
        all_y.append(d.y)
        all_p.append(yhat)
    y = np.concatenate(all_y)
    p = np.concatenate(all_p)
    out: Dict[str, Any] = {"pooled": _metrics(y, p), "per_cell": per_cell}
    if window_max_y is not None:
        m = y <= float(window_max_y)
        out["pooled_window"] = _metrics(y[m], p[m]) if m.sum() >= 3 else None
        out["window_max_y"] = float(window_max_y)
    return out


def restrict_to_window(datasets: Sequence[CellData], max_y: float) -> List[CellData]:
    """Truncate each cell's interval history at the first reference exceeding
    ``max_y``.

    Truncation (rather than masking) is required because the model is
    cumulative: dropping an interval from the middle would silently delete the
    stress that produced the later references.
    """
    out: List[CellData] = []
    for d in datasets:
        over = np.flatnonzero(d.y > float(max_y))
        n = int(over[0]) if over.size else d.n
        n = max(n, 3)
        out.append(CellData(
            cell=d.cell,
            duration_h=d.duration_h[:n],
            mean_soc=d.mean_soc[:n],
            mean_temperature_c=d.mean_temperature_c[:n],
            cyclic_mean_temperature_c=d.cyclic_mean_temperature_c[:n],
            dah=d.dah[:n],
            y=d.y[:n],
            cum_ah=d.cum_ah[:n],
            cum_duration_h=d.cum_duration_h[:n],
            t_end_h=d.t_end_h[:n],
        ))
    return out


def _metrics(y: np.ndarray, yhat: np.ndarray) -> Dict[str, float]:
    r = yhat - y
    ss_res = float(np.sum(r ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": int(y.size),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "rmse": float(np.sqrt(np.mean(r ** 2))),
        "rmse_pct_capacity": float(100.0 * np.sqrt(np.mean(r ** 2))),
        "mae": float(np.mean(np.abs(r))),
        "mae_pct_capacity": float(100.0 * np.mean(np.abs(r))),
        "bias": float(np.mean(r)),
        "bias_pct_capacity": float(100.0 * np.mean(r)),
        "max_abs_error": float(np.max(np.abs(r))),
        "y_range": [float(y.min()), float(y.max())],
    }


def fit(
    datasets: Sequence[CellData],
    *,
    layout: ParameterLayout,
    cfg: Dict[str, Any],
    q_nominal_ah: float,
    R: float,
    provenance: str = "pooled_all_cells",
    n_starts: Optional[int] = None,
    verbose: bool = True,
    fixed: Optional[Dict[int, float]] = None,
) -> FitResult:
    """Bounded multi-start trust-region least squares.

    ``fixed`` pins parameter indices to given values (used by the profile-
    likelihood scan over the calendar/cyclic split).
    """
    cal = cfg["calibration"]
    lo, hi = layout.bounds(cal["bounds"])
    n_starts = int(n_starts if n_starts is not None else cal["n_starts"])
    seed = int(cal["seed"])

    if fixed:
        lo = lo.copy()
        hi = hi.copy()
        for i, v in fixed.items():
            lo[i] = hi[i] = float(v)

    starts = [_anchor_start(layout, cal["anchor_start"], datasets, lo, hi)]
    if n_starts > 1:
        sampler = qmc.LatinHypercube(d=layout.size, seed=seed)
        u = sampler.random(n_starts - 1)
        starts.extend(list(lo + u * (hi - lo)))

    kw = dict(
        bounds=(lo, hi),
        method=str(cal["method"]),
        ftol=float(cal["ftol"]),
        xtol=float(cal["xtol"]),
        gtol=float(cal["gtol"]),
        max_nfev=int(cal["max_nfev"]),
        x_scale=cal.get("x_scale", "jac"),
    )
    args = (layout, list(datasets))
    kwargs = {"q_nominal_ah": q_nominal_ah, "R": R}

    if np.all(hi - lo <= 0):   # fully pinned
        kw["bounds"] = (lo - 1e-9, hi + 1e-9)

    best = None
    costs: List[float] = []
    with np.errstate(all="ignore"):
        for j, x0 in enumerate(starts):
            try:
                res = least_squares(
                    residuals, np.clip(x0, lo, hi), args=args, kwargs=kwargs, **kw,
                )
            except Exception as exc:  # pragma: no cover
                if verbose:
                    print(f"    start {j:>3}: failed ({exc})")
                continue
            costs.append(float(res.cost))
            if best is None or res.cost < best.cost:
                best = res
                if verbose:
                    print(f"    start {j:>3}: cost {res.cost:.6e}  <-- best so far")
    if best is None:
        raise RuntimeError("all calibration starts failed")

    se, corr, cond, flags = _uncertainty(best, layout, lo, hi, cfg)
    n_obs = int(sum(d.n for d in datasets))
    if layout.scale_cells:
        params: Any = CellParameters(
            layout, best.x, q_nominal_ah=q_nominal_ah, R=R, provenance=provenance,
        )
    else:
        params = layout.to_params(
            best.x, q_nominal_ah=q_nominal_ah, R=R, provenance=provenance,
        )
    return FitResult(
        params=params,
        layout=layout,
        x=np.asarray(best.x, dtype=np.float64),
        cost=float(best.cost),
        n_obs=n_obs,
        n_params=layout.size,
        success=bool(best.success),
        status=int(best.status),
        message=str(best.message),
        n_starts=len(costs),
        start_costs=costs,
        standard_errors=se,
        correlation=corr,
        jtj_condition=cond,
        flags=flags,
    )


def calendar_share(params: DegradationParameters, datasets: Sequence[CellData]) -> float:
    """Fraction of the end-of-record predicted loss attributed to calendar aging.

    The quantity whose identifiability decides whether the optimizer's
    time-vs-throughput trade-off is meaningful, so it is profiled explicitly.
    """
    cal = 0.0
    tot = 0.0
    for d in datasets:
        s = predict_cell_split(params_for(params, d.cell), d)
        cal += float(s["q_calendar"][-1])
        tot += float(s["q_total"][-1])
    return float(cal / tot) if tot > 0 else float("nan")


def profile_calendar_share(
    datasets: Sequence[CellData],
    *,
    layout: ParameterLayout,
    cfg: Dict[str, Any],
    q_nominal_ah: float,
    R: float,
    best: FitResult,
    n_points: int = 13,
    n_starts: int = 4,
    span_decades: float = 3.0,
) -> Dict[str, Any]:
    """Profile likelihood in the calendar rate coefficient: how well is the
    calendar/cyclic split determined?

    ``k_cal_ref`` is pinned across a grid and every other parameter re-fitted,
    which traces out cost against the resulting calendar share. A flat trace
    means the data cannot apportion fade between calendar and cyclic mechanisms
    and the apportionment must be reported as an assumption, not a finding.
    """
    if not layout.fit_calendar:
        raise ValueError(
            "profile_calendar_share requires a layout that fits the calendar "
            "branch; for the deployed anchored-calendar model use "
            "calendar_scale_sweep instead"
        )
    lo, hi = layout.bounds(cfg["calibration"]["bounds"])
    a_best = float(best.x[0])
    grid = np.linspace(
        max(a_best - span_decades, lo[0]), min(a_best + span_decades, hi[0]), int(n_points),
    )
    n_obs = int(sum(d.n for d in datasets))
    dof = max(n_obs - layout.size, 1)
    # F-based 95 % threshold on the residual sum of squares for 1 profiled parameter
    from scipy.stats import f as f_dist
    thresh = best.cost * (1.0 + f_dist.ppf(0.95, 1, dof) / dof)

    points: List[Dict[str, Any]] = []
    for a in grid:
        try:
            r = fit(
                datasets, layout=layout, cfg=cfg, q_nominal_ah=q_nominal_ah, R=R,
                provenance="profile", n_starts=n_starts, verbose=False,
                fixed={0: float(a)},
            )
        except Exception:
            continue
        gof = goodness_of_fit(r.params, datasets)
        points.append({
            "log10_k_cal_ref": float(a),
            "cost": r.cost,
            "calendar_share": calendar_share(r.params, datasets),
            "r2": gof["pooled"]["r2"],
            "rmse_pct_capacity": gof["pooled"]["rmse_pct_capacity"],
            "within_95pct_ci": bool(r.cost <= thresh),
        })

    inside = [p for p in points if p["within_95pct_ci"] and np.isfinite(p["calendar_share"])]
    shares = [p["calendar_share"] for p in inside]
    return {
        "profiled_parameter": "log10_k_cal_ref",
        "best_cost": best.cost,
        "cost_threshold_95pct": float(thresh),
        "n_points": len(points),
        "points": points,
        "best_calendar_share": calendar_share(best.params, datasets),
        "calendar_share_95pct_interval": (
            [float(min(shares)), float(max(shares))] if shares else None
        ),
        "identified": bool(shares and (max(shares) - min(shares)) < 0.25),
        "interpretation": (
            "A wide interval means the calendar/cyclic apportionment is not "
            "determined by this dataset (cumulative calendar time and cumulative "
            "throughput are collinear at r ~ 0.99 in the NASA RW duty cycle). "
            "The apportionment must then be treated as an assumption and its "
            "effect on protocol ranking checked by re-scoring, not asserted."
        ),
    }


def calendar_scale_sweep(
    datasets: Sequence[CellData],
    *,
    cfg: Dict[str, Any],
    q_nominal_ah: float,
    R: float,
    variant: str = "primary",
    n_starts: Optional[int] = None,
) -> Dict[str, Any]:
    """Refit the cyclic branch for each declared calendar-scale multiplier.

    The calendar branch is an assumption (NASA RW cannot identify it), so this
    quantifies how much of the measured fade the cyclic branch is asked to
    carry as that assumption is varied, and whether the calibration quality
    depends on it. ``scale = 0`` is the pure-cyclic limit.

    This is a calibration-side robustness check only. It reads no optimization
    output whatsoever (RULE 6); the protocol-ranking half of the sweep is run
    separately, downstream, against the frozen coefficients produced here.
    """
    multipliers = [
        float(m) for m in cfg["calendar_branch"]["sensitivity_multipliers"]
    ]
    window = float(cfg["calibration"]["operating_window_max_y"])
    cells = [d.cell for d in datasets]
    rows: List[Dict[str, Any]] = []
    for m in multipliers:
        layout = layouts_from_config(
            cfg, R=R, calendar_scale=m, cells=cells,
        )[variant]
        res = fit(
            datasets, layout=layout, cfg=cfg, q_nominal_ah=q_nominal_ah, R=R,
            provenance=f"calendar_scale={m:g}", n_starts=n_starts, verbose=False,
        )
        gof = goodness_of_fit(res.params, datasets, window_max_y=window)
        rep = params_for(res.params, cells[0])
        rows.append({
            "calendar_scale": m,
            "A_cal": rep.A_cal,
            "calendar_share": calendar_share(res.params, datasets),
            "r2": gof["pooled"]["r2"],
            "rmse_pct_capacity": gof["pooled"]["rmse_pct_capacity"],
            "mae_pct_capacity": gof["pooled"]["mae_pct_capacity"],
            "bias_pct_capacity": gof["pooled"]["bias_pct_capacity"],
            "p_c_rate": (
                float(res.x[layout.names.index("p_c_rate")])
                if "p_c_rate" in layout.names else None
            ),
            "Ea_cyc": float(rep.Ea_cyc[0]),
            "z_cyc": float(rep.z_cyc[0]),
            "n_flags": len(res.flags),
        })
    r2s = [r["r2"] for r in rows]
    return {
        "variant": variant,
        "multipliers": multipliers,
        "rows": rows,
        "r2_spread": float(max(r2s) - min(r2s)) if r2s else None,
        "interpretation": (
            "If calibration quality is nearly flat across the sweep, the "
            "measured fade does not distinguish these calendar assumptions and "
            "the cyclic branch simply absorbs the difference. That is a "
            "statement about this dataset, not a defect of the model — but it "
            "means the anchored calendar level must be carried as an "
            "assumption and its effect on protocol ranking checked downstream."
        ),
    }


def _uncertainty(
    res, layout: ParameterLayout, lo: np.ndarray, hi: np.ndarray, cfg: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float], List[str]]:
    """Jacobian-based standard errors, correlation matrix and flag list.

    Ill-conditioned ``JᵀJ`` is expected: Arrhenius prefactor/activation-energy
    pairs are near-collinear over the ~20 K span these cells span. The
    pseudo-inverse is used and the condition is reported rather than suppressed.
    """
    val = cfg["validation"]
    names = layout.names
    flags: List[str] = []

    span = np.maximum(hi - lo, 1e-12)
    frac = float(val["flag_bound_proximity_frac"])
    for i, nm in enumerate(names):
        if (res.x[i] - lo[i]) / span[i] < frac:
            flags.append(f"{nm}: at LOWER bound ({res.x[i]:.6g})")
        elif (hi[i] - res.x[i]) / span[i] < frac:
            flags.append(f"{nm}: at UPPER bound ({res.x[i]:.6g})")

    try:
        J = np.asarray(res.jac, dtype=np.float64)
        n, p = J.shape
        dof = max(n - p, 1)
        s2 = float(np.sum(res.fun ** 2)) / dof
        jtj = J.T @ J
        cond = float(np.linalg.cond(jtj))
        rank = int(np.linalg.matrix_rank(jtj, tol=1e-10 * max(np.abs(jtj).max(), 1e-30)))
        if rank < p:
            flags.append(
                f"JᵀJ numerical rank {rank} < {p} parameters — "
                f"{p - rank} direction(s) unconstrained by the data"
            )
        cov = s2 * np.linalg.pinv(jtj, rcond=1e-12)
        se = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
        denom = np.outer(se, se)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.where(denom > 0, cov / denom, np.nan)
        # A pseudo-inverse of a rank-deficient JᵀJ is not guaranteed PSD, so the
        # off-diagonals can exceed 1. Clip for reporting and rely on the rank
        # and condition-number flags above to signal the deficiency.
        corr = np.clip(corr, -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)
    except Exception:
        return None, None, None, flags + ["covariance unavailable (singular Jacobian)"]

    thr_rel = float(val["flag_relative_se"])
    for i, nm in enumerate(names):
        if abs(res.x[i]) > 1e-12 and se[i] / abs(res.x[i]) > thr_rel:
            flags.append(
                f"{nm}: weakly determined (SE/|θ| = {se[i] / abs(res.x[i]):.2f})"
            )

    thr_c = float(val["flag_abs_correlation"])
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            c = corr[i, j]
            if np.isfinite(c) and abs(c) > thr_c:
                flags.append(f"{names[i]} <-> {names[j]}: |corr| = {abs(c):.4f}")
    if cond > 1e12:
        flags.append(f"JᵀJ condition number {cond:.3e} — parameters not jointly identifiable")
    return se, corr, cond, flags


def datasets_from_dataframe(df, cells: Sequence[str], c_rate_bins: Sequence[float]) -> List[CellData]:
    from aacopt.calibration_data import cell_arrays

    out: List[CellData] = []
    for cell in cells:
        a = cell_arrays(df, cell, c_rate_bins)
        out.append(CellData(
            cell=cell,
            duration_h=a["duration_h"],
            mean_soc=a["mean_soc"],
            mean_temperature_c=a["mean_temperature_c"],
            cyclic_mean_temperature_c=a["cyclic_mean_temperature_c"],
            dah=a["dah"],
            y=a["y"],
            cum_ah=a["cum_ah"],
            cum_duration_h=a["cum_duration_h"],
            t_end_h=a["t_end_h"],
        ))
    return out
