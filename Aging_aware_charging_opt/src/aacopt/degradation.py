"""Hybrid calendar + cyclic degradation model.

Equations are unchanged from the study methodology; the coefficients are
re-derived in this project (``scripts/03_fit_degradation.py``) and loaded from
``configs/degradation_fitted.yaml``. This module hard-codes no fitted
coefficient.

Model
-----
::

    Q_cal(SOC, T, t_h) = A_cal * exp(B_cal*SOC)
                              * exp(-(Ea_cal + C_cal*SOC) / (R*T_K))
                              * t_h ** z_cal

    Q_cyc(Ah, T, C)    = B_cyc(C) * exp(-Ea_cyc(C) / (R*T_K)) * Ah ** z_cyc(C)

    Q_total            = Q_cal + Q_cyc

The cyclic coefficients live on a C-rate grid (default ``{0.5, 1.0, 2.0}``).
Throughput at an arbitrary C-rate is split between the two neighbouring grid
nodes with linear ("tent") weights, clamped outside the grid, and each node
carries its own accumulator. At a grid node this is exactly the single-rate
equation above; across a duty cycle spanning several C-rates it is the natural
mixture, ``Q_cyc = sum_b Q_cyc(Ah_b, T, C_b)``. The weighting is continuous in
C, so the objective surface the optimizer sees has no bin-boundary steps.

Accumulation under time-varying stress
--------------------------------------
A sub-linear power law cannot be advanced by summing increments evaluated at
different stresses. Equivalent-time (state-shift) integration is used:

    t_eq = (Q/k)**(1/z)          Q <- k * (t_eq + delta) ** z

Substituting ``u = Q**(1/z)`` turns this recursion into a plain sum, which is
both the exact solution and the reason the fit is cheap:

    u_m = u_{m-1} + k_m**(1/z) * delta_m        Q_m = u_m ** z

so ``Q_M = ( sum_m k_m**(1/z) * delta_m ) ** z``. Properties: exact reduction
to the closed form under constant stress, continuity when the stress changes,
monotone non-decreasing, and independent of the order in which stresses are
applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

R_UNIVERSAL = 8.314          # J mol^-1 K^-1
KELVIN_OFFSET = 273.15
MIN_KELVIN = 200.0
Z_FLOOR = 1e-6


@dataclass(frozen=True)
class DegradationParameters:
    """Calibrated physical coefficients. No reward weights live here."""

    A_cal: float
    B_cal: float
    C_cal: float
    Ea_cal: float
    z_cal: float
    c_rate_bins: Tuple[float, ...]
    B_cyc: Tuple[float, ...]
    Ea_cyc: Tuple[float, ...]
    z_cyc: Tuple[float, ...]
    R: float = R_UNIVERSAL
    q_nominal_ah: float = 2.2
    provenance: str = "unspecified"

    def __post_init__(self) -> None:
        n = len(self.c_rate_bins)
        for name in ("B_cyc", "Ea_cyc", "z_cyc"):
            if len(getattr(self, name)) != n:
                raise ValueError(
                    f"{name} has {len(getattr(self, name))} entries, expected {n}"
                )
        if list(self.c_rate_bins) != sorted(self.c_rate_bins):
            raise ValueError("c_rate_bins must be ascending")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "A_cal": float(self.A_cal),
            "B_cal": float(self.B_cal),
            "C_cal": float(self.C_cal),
            "Ea_cal": float(self.Ea_cal),
            "z_cal": float(self.z_cal),
            "c_rate_bins": [float(x) for x in self.c_rate_bins],
            "B_cyc": [float(x) for x in self.B_cyc],
            "Ea_cyc": [float(x) for x in self.Ea_cyc],
            "z_cyc": [float(x) for x in self.z_cyc],
            "R": float(self.R),
            "q_nominal_ah": float(self.q_nominal_ah),
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DegradationParameters":
        return cls(
            A_cal=float(d["A_cal"]),
            B_cal=float(d["B_cal"]),
            C_cal=float(d["C_cal"]),
            Ea_cal=float(d["Ea_cal"]),
            z_cal=float(d["z_cal"]),
            c_rate_bins=tuple(float(x) for x in d["c_rate_bins"]),
            B_cyc=tuple(float(x) for x in d["B_cyc"]),
            Ea_cyc=tuple(float(x) for x in d["Ea_cyc"]),
            z_cyc=tuple(float(x) for x in d["z_cyc"]),
            R=float(d.get("R", R_UNIVERSAL)),
            q_nominal_ah=float(d.get("q_nominal_ah", 2.2)),
            provenance=str(d.get("provenance", "unspecified")),
        )

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "DegradationParameters":
        """Load the deployed (pooled all-cells) fit."""
        import yaml

        from aacopt.config import config_path

        p = Path(path) if path is not None else config_path("degradation_fitted")
        if not p.is_file():
            raise FileNotFoundError(
                f"{p} not found — run scripts/03_fit_degradation.py first. "
                "This project deliberately ships no pre-set coefficients."
            )
        blob = yaml.safe_load(p.read_text(encoding="utf-8"))
        deployed = blob["deployed"]
        if "per_cell" in deployed:
            # leftover from a discarded per-cell parameterization
            first = next(iter(deployed["per_cell"].values()))
            return cls.from_dict(first)
        return cls.from_dict(deployed)


def calendar_from_anchors(
    anchors: Dict[str, Any], *, R: float = R_UNIVERSAL, scale: float = 1.0,
) -> Dict[str, float]:
    """Derive the calendar coefficients from interpretable literature anchors.

    NASA RW cannot identify the calendar branch (no storage arm; ~9 % rest, all
    at low SOC; elapsed time collinear with throughput at r ~ 0.99), so the
    branch is fixed from published calendar-aging results instead of fitted —
    see the ``calendar_branch`` block of ``configs/degradation.yaml`` for the
    disclosure and citations.

    The anchors are checkable physical statements rather than coefficients:

    ``z_cal``
        Power-law exponent (0.5 = SEI-diffusion-limited sqrt-of-time).
    ``Ea_cal_j_per_mol``
        Apparent activation energy of calendar fade.
    ``soc_acceleration_ratio``
        Rate ratio between SOC = 1 and SOC = 0; ``B_cal = ln(ratio)``.
    ``fade_fraction_at_anchor`` at (``anchor_time_h``, ``anchor_soc``,
    ``anchor_temperature_k``)
        Pins the remaining scale degree of freedom, so ``A_cal`` is *solved
        for* rather than asserted.

    ``scale`` multiplies the anchored fade level and exists for the declared
    sensitivity sweep — it is the knob the downstream robustness check turns,
    and ``scale = 0`` gives the pure-cyclic limit.
    """
    z_cal = float(anchors["z_cal"])
    ea_cal = float(anchors["Ea_cal_j_per_mol"])
    c_cal = float(anchors["C_cal"])
    ratio = float(anchors["soc_acceleration_ratio"])
    if ratio <= 0.0:
        raise ValueError("soc_acceleration_ratio must be positive")
    b_cal = float(np.log(ratio))

    y_anchor = float(anchors["fade_fraction_at_anchor"]) * float(scale)
    t_h = float(anchors["anchor_time_h"])
    soc = float(anchors["anchor_soc"])
    t_k = float(anchors["anchor_temperature_k"])
    if t_h <= 0.0:
        raise ValueError("anchor_time_h must be positive")

    shape = (
        np.exp(b_cal * soc)
        * np.exp(-(ea_cal + c_cal * soc) / (R * t_k))
        * t_h ** z_cal
    )
    a_cal = 0.0 if y_anchor <= 0.0 else float(y_anchor / shape)
    return {
        "A_cal": a_cal,
        "B_cal": b_cal,
        "C_cal": c_cal,
        "Ea_cal": ea_cal,
        "z_cal": z_cal,
    }


def kelvin(t_c) -> np.ndarray:
    return np.maximum(np.asarray(t_c, dtype=np.float64) + KELVIN_OFFSET, MIN_KELVIN)


def tent_weights(c_rate, bins: Sequence[float]) -> np.ndarray:
    """Linear split of throughput between neighbouring C-rate grid nodes.

    Returns shape ``(..., n_bins)`` rows summing to 1. Continuous in ``c_rate``;
    clamped to the end node outside the grid.
    """
    c = np.abs(np.asarray(c_rate, dtype=np.float64))
    g = np.asarray(bins, dtype=np.float64)
    n = g.size
    w = np.zeros(c.shape + (n,), dtype=np.float64)
    if n == 1:
        w[..., 0] = 1.0
        return w
    cc = np.clip(c, g[0], g[-1])
    hi = np.clip(np.searchsorted(g, cc, side="left"), 1, n - 1)
    lo = hi - 1
    span = g[hi] - g[lo]
    frac = np.where(span > 0, (cc - g[lo]) / span, 0.0)
    idx = np.indices(c.shape)
    w[(*idx, lo)] = 1.0 - frac
    w[(*idx, hi)] += frac
    return w


def bin_throughput(
    c_rate, ah, bins: Sequence[float],
) -> np.ndarray:
    """Sum ``ah`` into C-rate bins using tent weights. Returns ``(n_bins,)``."""
    c = np.asarray(c_rate, dtype=np.float64).ravel()
    a = np.asarray(ah, dtype=np.float64).ravel()
    if c.size == 0:
        return np.zeros(len(bins))
    return (tent_weights(c, bins) * a[:, None]).sum(axis=0)


class HybridDegradationModel:
    """Evaluate the calibrated model: closed form, cumulative, or per-session."""

    def __init__(self, params: DegradationParameters):
        self.p = params
        self._bins = np.asarray(params.c_rate_bins, dtype=np.float64)
        self._B = np.asarray(params.B_cyc, dtype=np.float64)
        self._Ea = np.asarray(params.Ea_cyc, dtype=np.float64)
        self._z = np.maximum(np.asarray(params.z_cyc, dtype=np.float64), Z_FLOOR)
        self._z_cal = max(float(params.z_cal), Z_FLOOR)

    # ── stress-dependent rate coefficients ──────────────────────────────────

    def calendar_k(self, soc, t_c) -> np.ndarray:
        """k in Q_cal = k * t_h**z_cal."""
        p = self.p
        s = np.clip(np.asarray(soc, dtype=np.float64), 0.0, 1.0)
        return (
            p.A_cal
            * np.exp(p.B_cal * s)
            * np.exp(-(p.Ea_cal + p.C_cal * s) / (p.R * kelvin(t_c)))
        )

    def cyclic_k_bins(self, t_c) -> np.ndarray:
        """k per grid node in Q_cyc,b = k_b * Ah_b**z_b. Shape ``(..., n_bins)``."""
        tk = kelvin(t_c)
        return self._B * np.exp(-self._Ea / (self.p.R * np.atleast_1d(tk)[..., None]))

    def cyclic_coeffs_at(self, c_rate) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Coefficients linearly interpolated to an arbitrary C-rate (reporting)."""
        c = np.abs(np.asarray(c_rate, dtype=np.float64))
        return (
            np.interp(c, self._bins, self._B),
            np.interp(c, self._bins, self._Ea),
            np.interp(c, self._bins, self._z),
        )

    # ── closed forms (constant stress) ──────────────────────────────────────

    def q_calendar(self, soc, t_c, t_hours) -> np.ndarray:
        t_h = np.maximum(np.asarray(t_hours, dtype=np.float64), 0.0)
        return self.calendar_k(soc, t_c) * np.power(t_h, self._z_cal)

    def q_cyclic_binned(self, ah_bins, t_c) -> np.ndarray:
        """Q_cyc from per-bin throughput. ``ah_bins`` shape ``(..., n_bins)``."""
        a = np.maximum(np.asarray(ah_bins, dtype=np.float64), 0.0)
        k = self.cyclic_k_bins(t_c)
        return np.sum(k * np.power(a, self._z), axis=-1)

    def q_cyclic(self, ah, t_c, c_rate) -> np.ndarray:
        """Q_cyc for throughput ``ah`` all delivered at a single C-rate."""
        a = np.atleast_1d(np.asarray(ah, dtype=np.float64))
        w = tent_weights(np.atleast_1d(np.asarray(c_rate, dtype=np.float64)), self._bins)
        out = self.q_cyclic_binned(w * a[..., None], t_c)
        return out if np.ndim(ah) else float(np.asarray(out).ravel()[0])

    def session_closed_form(
        self,
        *,
        current_a: np.ndarray,
        temperature_c: np.ndarray,
        soc: np.ndarray,
        dt_s: float = 1.0,
        rest_current_threshold_a: float = 0.01,
    ) -> Dict[str, float]:
        """Paper session-mean closed form (Eqs. calendar + cyclic).

        Calendar uses full-session mean SOC / T and elapsed hours.
        Cyclic uses charge-only Ah and mean charge C-rate, with the calibrated
        C-rate grid (tent interpolation of B_cyc, Ea_cyc, z_cyc).
        """
        i = np.asarray(current_a, dtype=np.float64)
        t_c = np.asarray(temperature_c, dtype=np.float64)
        s = np.asarray(soc, dtype=np.float64)
        n = int(i.size)
        duration_h = n * float(dt_s) / 3600.0
        charging = i < -abs(float(rest_current_threshold_a))
        q_ah = float(self.p.q_nominal_ah)
        if charging.any():
            ah = float(np.sum(np.abs(i[charging])) * float(dt_s) / 3600.0)
            c_rate = float(np.mean(np.abs(i[charging])) / max(q_ah, 1e-12))
        else:
            ah, c_rate = 0.0, 0.0
        mean_soc = float(s.mean()) if s.size else 0.0
        mean_t = float(t_c.mean()) if t_c.size else 25.0
        q_cal = float(np.asarray(self.q_calendar(mean_soc, mean_t, duration_h)).ravel()[0])
        q_cyc = float(self.q_cyclic(ah, mean_t, c_rate))
        return {
            "duration_h": duration_h,
            "mean_soc": mean_soc,
            "mean_temperature_c": mean_t,
            "ah_throughput": ah,
            "nominal_c_rate": c_rate,
            "delta_soc": float(s[-1] - s[0]) if s.size else 0.0,
            "q_calendar": q_cal,
            "q_cyclic": q_cyc,
            "q_total": q_cal + q_cyc,
        }

    # ── cumulative accumulation (calibration + lifetime) ────────────────────

    def accumulate_intervals(
        self,
        *,
        duration_h: np.ndarray,
        mean_soc: np.ndarray,
        mean_temperature_c: np.ndarray,
        dah: np.ndarray,
        temperature_c_cyclic: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Cumulative ``Q_total`` after each interval (vectorized exact form).

        ``dah`` has shape ``(n_intervals, n_bins)``. ``temperature_c_cyclic``
        optionally supplies a throughput-weighted temperature per interval for
        the cyclic term (falls back to ``mean_temperature_c``).
        """
        s = self.cumulative_split(
            duration_h=duration_h,
            mean_soc=mean_soc,
            mean_temperature_c=mean_temperature_c,
            dah=dah,
            temperature_c_cyclic=temperature_c_cyclic,
        )
        return s["q_total"]

    def cumulative_split(
        self,
        *,
        duration_h: np.ndarray,
        mean_soc: np.ndarray,
        mean_temperature_c: np.ndarray,
        dah: np.ndarray,
        temperature_c_cyclic: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        dt = np.maximum(np.asarray(duration_h, dtype=np.float64), 0.0)
        soc = np.asarray(mean_soc, dtype=np.float64)
        t_cal = np.asarray(mean_temperature_c, dtype=np.float64)
        t_cyc = t_cal if temperature_c_cyclic is None else np.asarray(
            temperature_c_cyclic, dtype=np.float64,
        )
        d = np.maximum(np.asarray(dah, dtype=np.float64), 0.0)

        z_cal = self._z_cal
        u_cal = np.cumsum(np.power(self.calendar_k(soc, t_cal), 1.0 / z_cal) * dt)
        q_cal = np.power(u_cal, z_cal)

        k_cyc = self.cyclic_k_bins(t_cyc)              # (n_intervals, n_bins)
        u_cyc = np.cumsum(np.power(k_cyc, 1.0 / self._z) * d, axis=0)
        q_cyc = np.sum(np.power(u_cyc, self._z), axis=1)

        q_cal = np.where(np.isfinite(q_cal), q_cal, np.inf)
        q_cyc = np.where(np.isfinite(q_cyc), q_cyc, np.inf)
        return {"q_calendar": q_cal, "q_cyclic": q_cyc, "q_total": q_cal + q_cyc}

    def new_state(self) -> "DegradationState":
        return DegradationState(model=self)


@dataclass
class DegradationState:
    """Accumulation state in equivalent-exposure (``u = Q**(1/z)``) coordinates."""

    model: HybridDegradationModel
    u_cal: float = 0.0
    u_cyc: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.u_cyc is None:
            self.u_cyc = np.zeros(self.model._bins.size, dtype=np.float64)

    @property
    def q_calendar(self) -> float:
        return float(self.u_cal ** self.model._z_cal)

    @property
    def q_cyclic(self) -> float:
        return float(np.sum(np.power(self.u_cyc, self.model._z)))

    @property
    def q_total(self) -> float:
        return self.q_calendar + self.q_cyclic

    def advance_calendar(self, soc, t_c, dt_h) -> None:
        """Vectorized over arrays of samples sharing one dt."""
        k = self.model.calendar_k(soc, t_c)
        z = self.model._z_cal
        self.u_cal += float(np.sum(np.power(k, 1.0 / z) * np.asarray(dt_h)))

    def advance_cyclic_binned(self, ah_bins_weighted: np.ndarray, t_c) -> None:
        """``ah_bins_weighted`` shape ``(n_samples, n_bins)`` with matching ``t_c``."""
        a = np.asarray(ah_bins_weighted, dtype=np.float64)
        if a.size == 0:
            return
        k = self.model.cyclic_k_bins(np.atleast_1d(t_c))
        self.u_cyc = self.u_cyc + np.sum(np.power(k, 1.0 / self.model._z) * a, axis=0)

    def copy(self) -> "DegradationState":
        return DegradationState(self.model, float(self.u_cal), np.array(self.u_cyc))


# ── session-level evaluation (used by the reward, baselines and lifetime) ───


def session_stress_summary(
    *,
    current_a: np.ndarray,
    temperature_c: np.ndarray,
    soc: np.ndarray,
    dt_s: float,
    q_nominal_ah: float,
    c_rate_bins: Sequence[float],
    rest_current_threshold_a: float = 0.01,
) -> Dict[str, Any]:
    """Reduce a simulated trajectory to the stress descriptors the model needs.

    Binned exactly as the calibration data was (tent weights on the same
    C-rate grid). Sign convention: negative current = charge.
    """
    i = np.asarray(current_a, dtype=np.float64)
    t_c = np.asarray(temperature_c, dtype=np.float64)
    s = np.asarray(soc, dtype=np.float64)
    n = int(i.size)
    n_bins = len(c_rate_bins)
    if n == 0:
        return {
            "duration_h": 0.0, "duration_s": 0.0, "rest_h": 0.0, "charge_h": 0.0,
            "mean_soc": 0.0, "mean_temperature_c": 25.0, "max_temperature_c": 25.0,
            "dah": np.zeros(n_bins), "ah_charge_total": 0.0,
            "mean_charge_c_rate": 0.0, "max_charge_c_rate": 0.0,
            "cyclic_mean_temperature_c": 25.0, "delta_soc": 0.0,
        }

    dt_h = float(dt_s) / 3600.0
    charging = i < -abs(rest_current_threshold_a)
    i_abs = np.abs(i)
    c_rate = i_abs / float(q_nominal_ah)
    ah = i_abs * dt_h

    dah = np.zeros(n_bins)
    cyc_t = float(t_c.mean()) if t_c.size else 25.0
    if charging.any():
        dah = bin_throughput(c_rate[charging], ah[charging], c_rate_bins)
        w = ah[charging]
        if w.sum() > 0:
            cyc_t = float(np.sum(t_c[charging] * w) / w.sum())

    return {
        "duration_h": n * dt_h,
        "duration_s": n * float(dt_s),
        "rest_h": float(np.count_nonzero(~charging)) * dt_h,
        "charge_h": float(np.count_nonzero(charging)) * dt_h,
        "mean_soc": float(s.mean()) if s.size else 0.0,
        "mean_temperature_c": float(t_c.mean()) if t_c.size else 25.0,
        "max_temperature_c": float(t_c.max()) if t_c.size else 25.0,
        "cyclic_mean_temperature_c": cyc_t,
        "dah": dah,
        "ah_charge_total": float(dah.sum()),
        "mean_charge_c_rate": float(c_rate[charging].mean()) if charging.any() else 0.0,
        "max_charge_c_rate": float(c_rate[charging].max()) if charging.any() else 0.0,
        "delta_soc": float(s[-1] - s[0]) if s.size else 0.0,
    }


def session_degradation(
    model: HybridDegradationModel,
    *,
    current_a: np.ndarray,
    temperature_c: np.ndarray,
    soc: np.ndarray,
    dt_s: float,
    rest_current_threshold_a: float = 0.01,
    state: Optional[DegradationState] = None,
) -> Dict[str, Any]:
    """Q_cal / Q_cyc accrued by one charging session, integrated sample by sample.

    The actual SOC / temperature / C-rate path is respected rather than session
    means. Passing ``state`` appends the session to an existing exposure history
    (used by the equivalent-cycle projection), so the returned ``q_*`` are the
    *increments* contributed by this session at that point in the history.
    """
    st = state.copy() if state is not None else model.new_state()
    cal0, cyc0, tot0 = st.q_calendar, st.q_cyclic, st.q_total

    i = np.asarray(current_a, dtype=np.float64)
    t_c = np.asarray(temperature_c, dtype=np.float64)
    s = np.asarray(soc, dtype=np.float64)
    dt_h = float(dt_s) / 3600.0
    q_nom = model.p.q_nominal_ah

    if i.size:
        st.advance_calendar(s, t_c, np.full(i.size, dt_h))
        charging = i < -abs(rest_current_threshold_a)
        if charging.any():
            i_abs = np.abs(i[charging])
            w = tent_weights(i_abs / q_nom, model.p.c_rate_bins)
            st.advance_cyclic_binned(w * (i_abs * dt_h)[:, None], t_c[charging])

    summary = session_stress_summary(
        current_a=i, temperature_c=t_c, soc=s, dt_s=dt_s,
        q_nominal_ah=q_nom, c_rate_bins=model.p.c_rate_bins,
        rest_current_threshold_a=rest_current_threshold_a,
    )
    return {
        **summary,
        "dah": [float(x) for x in np.asarray(summary["dah"]).ravel()],
        "q_calendar": st.q_calendar - cal0,
        "q_cyclic": st.q_cyclic - cyc0,
        "q_total": st.q_total - tot0,
        "state": st,
    }


def project_repeated_sessions(
    model: HybridDegradationModel,
    *,
    dah: Sequence[float],
    duration_h: float,
    mean_soc: float,
    mean_temperature_c: float,
    cyclic_mean_temperature_c: Optional[float] = None,
    n_cycles: int,
) -> Dict[str, np.ndarray]:
    """Relative capacity-retention projection over ``n_cycles`` identical sessions.

    Uses the exact equivalent-exposure form, so this is the closed-form
    ``Q(N) = (N * k**(1/z) * delta) ** z`` per accumulator — no loop needed.
    """
    n = np.arange(0, int(n_cycles) + 1, dtype=np.float64)
    t_cyc = (
        mean_temperature_c if cyclic_mean_temperature_c is None
        else cyclic_mean_temperature_c
    )
    z_cal = model._z_cal
    k_cal = float(model.calendar_k(mean_soc, mean_temperature_c))
    u_cal = n * (k_cal ** (1.0 / z_cal)) * max(float(duration_h), 0.0)
    q_cal = np.power(u_cal, z_cal)

    k_cyc = np.asarray(model.cyclic_k_bins(t_cyc), dtype=np.float64).ravel()
    d = np.maximum(np.asarray(dah, dtype=np.float64).ravel(), 0.0)
    u_cyc = n[:, None] * np.power(k_cyc, 1.0 / model._z) * d
    q_cyc = np.sum(np.power(u_cyc, model._z), axis=1)

    q_tot = q_cal + q_cyc
    return {
        "cycles": n,
        "cum_ah": n * float(d.sum()),
        "q_calendar": q_cal,
        "q_cyclic": q_cyc,
        "q_total": q_tot,
        "retention_pct": np.clip(100.0 * (1.0 - q_tot), 0.0, 100.0),
    }
