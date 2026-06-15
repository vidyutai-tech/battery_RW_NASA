"""
Priority 3 — Pareto analysis over feasible BO candidates.

Objectives (all minimized):
  - duration_min
  - sei_per_pct_soc
  - voltage_stress_v2_min
  - temperature_penalty_c2_min

Extracts candidates from family optimization history, builds non-dominated
sets, and tags Fastest / Lifetime / Balanced reference profiles.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

PARETO_OBJECTIVES = (
    "duration_min",
    "sei_per_pct_soc",
    "voltage_stress_v2_min",
    "temperature_penalty_c2_min",
)

PHYSICS_PARETO_OBJECTIVES = (
    "duration_min",
    "capacity_fade_pct",
    "voltage_stress_v2_min",
    "temperature_penalty_c2_min",
)

DEGRADATION_LABELS = {
    "sei_per_pct_soc": "SEI / ΔSoC",
    # Plot axis: short. Wang (2011) ΔQ/Q₀ from BDT session — see physics_degradation.py
    "capacity_fade_pct": "ΔQ/Q₀ [%]",
    "equiv_cycles_to_eol": "N to EOL",
}


def resolve_pareto_config(constraints: Optional[Mapping] = None) -> tuple[tuple[str, ...], str, str]:
    """
    Return (4-objective tuple, 2D plot y-attribute, y-axis label).

    Physics BO runs use capacity_fade_pct instead of SEI/ΔSoC on Pareto axes.
    """
    mode = (constraints or {}).get("objective_mode", "composite")
    if mode == "physics":
        return PHYSICS_PARETO_OBJECTIVES, "capacity_fade_pct", DEGRADATION_LABELS["capacity_fade_pct"]
    return PARETO_OBJECTIVES, "sei_per_pct_soc", DEGRADATION_LABELS["sei_per_pct_soc"]


def degradation_value(metrics: Mapping, key: str) -> float:
    if key == "capacity_fade_pct":
        val = metrics.get("capacity_fade_pct")
        if val is not None:
            return float(val)
    return float(metrics.get("sei_per_pct_soc", float("nan")))


def format_degradation_value(value: float, key: str) -> str:
    import math

    if not math.isfinite(value):
        return "nan"
    if key == "capacity_fade_pct":
        return f"{value:.3f}"
    return f"{value:.1f}"


def degradation_summary(
    metrics: Mapping,
    constraints: Optional[Mapping] = None,
) -> str:
    """Objective-aware degradation string for logs and comparison tables."""
    if constraints is None:
        comp = metrics.get("components") or {}
        if comp.get("objective_mode") == "physics_degradation":
            constraints = {"objective_mode": "physics"}
        else:
            constraints = {"objective_mode": "composite"}
    _, key, label = resolve_pareto_config(constraints)
    val = degradation_value(metrics, key)
    return f"{label}={format_degradation_value(val, key)}"


@dataclass
class ParetoCandidate:
    family_id: str
    family_label: str
    params: Dict
    duration_min: float
    sei_per_pct_soc: float
    voltage_stress_v2_min: float
    temperature_penalty_c2_min: float
    loss: float
    peak_voltage: float
    peak_temperature: float
    end_reason: str
    capacity_fade_pct: Optional[float] = None
    equiv_cycles_to_eol: Optional[float] = None

    def objective_vector(self, objectives: Sequence[str] = PARETO_OBJECTIVES) -> List[float]:
        out: List[float] = []
        for key in objectives:
            val = getattr(self, key, None)
            if val is None:
                if key == "capacity_fade_pct":
                    val = self.sei_per_pct_soc
                else:
                    val = 0.0
            out.append(float(val))
        return out

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TaggedProfiles:
    fastest: Optional[ParetoCandidate] = None
    lifetime: Optional[ParetoCandidate] = None
    balanced: Optional[ParetoCandidate] = None

    def to_dict(self) -> Dict:
        return {
            "fastest": self.fastest.to_dict() if self.fastest else None,
            "lifetime": self.lifetime.to_dict() if self.lifetime else None,
            "balanced": self.balanced.to_dict() if self.balanced else None,
        }


@dataclass
class ParetoAnalysisResult:
    objectives: List[str] = field(default_factory=lambda: list(PARETO_OBJECTIVES))
    degradation_key: str = "sei_per_pct_soc"
    degradation_label: str = DEGRADATION_LABELS["sei_per_pct_soc"]
    n_feasible_total: int = 0
    n_pareto_global: int = 0
    pareto_front: List[ParetoCandidate] = field(default_factory=list)
    tagged_global: TaggedProfiles = field(default_factory=TaggedProfiles)
    per_family: Dict[str, Dict] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "objectives": self.objectives,
            "degradation_key": self.degradation_key,
            "degradation_label": self.degradation_label,
            "n_feasible_total": self.n_feasible_total,
            "n_pareto_global": self.n_pareto_global,
            "pareto_front": [c.to_dict() for c in self.pareto_front],
            "tagged_global": self.tagged_global.to_dict(),
            "per_family": self.per_family,
        }


def _metric_float(metrics: Mapping, key: str, default: float = 0.0) -> float:
    val = metrics.get(key, default)
    try:
        out = float(val)
    except (TypeError, ValueError):
        return default
    return out if out == out else default  # NaN → default


def candidate_from_history_entry(
    entry: Mapping,
    *,
    family_id: str,
    family_label: str,
) -> Optional[ParetoCandidate]:
    metrics = entry.get("metrics") or {}
    if not metrics.get("feasible"):
        return None
    sei = metrics.get("sei_per_pct_soc")
    fade = metrics.get("capacity_fade_pct")
    if sei is None and fade is None:
        return None
    return ParetoCandidate(
        family_id=family_id,
        family_label=family_label,
        params=dict(entry.get("params") or {}),
        duration_min=_metric_float(metrics, "duration_min"),
        sei_per_pct_soc=float(sei) if sei is not None else float("nan"),
        voltage_stress_v2_min=_metric_float(metrics, "voltage_stress_v2_min"),
        temperature_penalty_c2_min=_metric_float(metrics, "temperature_penalty_c2_min"),
        capacity_fade_pct=float(fade) if fade is not None else None,
        equiv_cycles_to_eol=(
            float(metrics["equiv_cycles_to_eol"])
            if metrics.get("equiv_cycles_to_eol") is not None
            else None
        ),
        loss=_metric_float(metrics, "loss", _metric_float(entry, "loss")),
        peak_voltage=_metric_float(metrics, "peak_voltage"),
        peak_temperature=_metric_float(metrics, "peak_temperature"),
        end_reason=str(metrics.get("end_reason", entry.get("end_reason", ""))),
    )


def extract_feasible_candidates(
    families_data: Mapping[str, Mapping],
    *,
    degradation_key: str = "sei_per_pct_soc",
) -> List[ParetoCandidate]:
    """Collect unique feasible candidates from each family's BO history."""
    seen: set[tuple] = set()
    out: List[ParetoCandidate] = []

    for fid, fam in families_data.items():
        label = str(fam.get("family_label", fid))
        history = fam.get("history") or []

        entries = list(history)
        best_params = fam.get("best_params")
        best_metrics = fam.get("best_metrics")
        if best_params and best_metrics and best_metrics.get("feasible"):
            entries.append({
                "params": best_params,
                "metrics": best_metrics,
                "loss": fam.get("best_loss"),
            })

        for entry in entries:
            cand = candidate_from_history_entry(entry, family_id=fid, family_label=label)
            if cand is None:
                continue
            deg_val = getattr(cand, degradation_key, None)
            if deg_val is None:
                deg_val = cand.sei_per_pct_soc
            key = (
                fid,
                round(cand.duration_min, 3),
                round(float(deg_val), 4),
                round(cand.voltage_stress_v2_min, 4),
                tuple(sorted((k, round(float(v), 6)) for k, v in cand.params.items()
                              if k != "family_id" and isinstance(v, (int, float)))),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(cand)
    return out


def dominates(
    a: ParetoCandidate,
    b: ParetoCandidate,
    objectives: Sequence[str] = PARETO_OBJECTIVES,
) -> bool:
    """True if *a* is no worse on all objectives and strictly better on at least one."""
    av = a.objective_vector(objectives)
    bv = b.objective_vector(objectives)
    if any(x > y for x, y in zip(av, bv)):
        return False
    return any(x < y for x, y in zip(av, bv))


def pareto_front(
    candidates: Sequence[ParetoCandidate],
    objectives: Sequence[str] = PARETO_OBJECTIVES,
    *,
    degradation_key: str = "sei_per_pct_soc",
) -> List[ParetoCandidate]:
    if not candidates:
        return []
    front: List[ParetoCandidate] = []
    for i, a in enumerate(candidates):
        if any(i != j and dominates(b, a, objectives) for j, b in enumerate(candidates)):
            continue
        front.append(a)

    def _sort_key(c: ParetoCandidate) -> tuple:
        deg = getattr(c, degradation_key, None)
        if deg is None:
            deg = c.sei_per_pct_soc
        return (c.duration_min, float(deg))

    return sorted(front, key=_sort_key)


def select_fastest(
    candidates: Sequence[ParetoCandidate],
    *,
    degradation_key: str = "sei_per_pct_soc",
) -> Optional[ParetoCandidate]:
    if not candidates:
        return None

    def _key(c: ParetoCandidate) -> tuple:
        deg = getattr(c, degradation_key, None)
        if deg is None:
            deg = c.sei_per_pct_soc
        return (c.duration_min, float(deg))

    return min(candidates, key=_key)


def select_lifetime(
    candidates: Sequence[ParetoCandidate],
    *,
    degradation_key: str = "sei_per_pct_soc",
) -> Optional[ParetoCandidate]:
    if not candidates:
        return None

    def _key(c: ParetoCandidate) -> tuple:
        deg = getattr(c, degradation_key, None)
        if deg is None:
            deg = c.sei_per_pct_soc
        return (float(deg), c.duration_min)

    return min(candidates, key=_key)


def select_balanced(
    front: Sequence[ParetoCandidate],
    objectives: Sequence[str] = PARETO_OBJECTIVES,
) -> Optional[ParetoCandidate]:
    """
    Knee / compromise point: minimum normalized distance to the utopia point
    (best value per objective on the supplied set, usually the Pareto front).
    """
    if not front:
        return None
    if len(front) == 1:
        return front[0]

    def _metric(c: ParetoCandidate, key: str) -> float:
        val = getattr(c, key, None)
        if val is None or (isinstance(val, float) and val != val):
            if key == "capacity_fade_pct":
                return float(c.sei_per_pct_soc)
            return 0.0
        return float(val)

    utopia = {k: min(_metric(c, k) for c in front) for k in objectives}
    span = {
        k: max(_metric(c, k) for c in front) - utopia[k] + 1e-9
        for k in objectives
    }

    def _score(c: ParetoCandidate) -> float:
        return sum(
            ((_metric(c, k) - utopia[k]) / span[k]) ** 2
            for k in objectives
        )

    return min(front, key=_score)


def tag_profiles(
    all_feasible: Sequence[ParetoCandidate],
    front: Sequence[ParetoCandidate],
    *,
    objectives: Sequence[str] = PARETO_OBJECTIVES,
    degradation_key: str = "sei_per_pct_soc",
) -> TaggedProfiles:
    """
    Fastest / Lifetime from all feasible candidates; Balanced from the Pareto front.
    """
    return TaggedProfiles(
        fastest=select_fastest(all_feasible, degradation_key=degradation_key),
        lifetime=select_lifetime(all_feasible, degradation_key=degradation_key),
        balanced=select_balanced(front, objectives),
    )


def analyze_family_results(
    families_data: Mapping[str, Mapping],
    *,
    constraints: Optional[Mapping] = None,
) -> ParetoAnalysisResult:
    objectives, degradation_key, degradation_label = resolve_pareto_config(constraints)
    obj_list = list(objectives)
    all_feasible = extract_feasible_candidates(
        families_data, degradation_key=degradation_key,
    )
    global_front = pareto_front(
        all_feasible, obj_list, degradation_key=degradation_key,
    )

    per_family: Dict[str, Dict] = {}
    for fid in families_data:
        fam_cands = [c for c in all_feasible if c.family_id == fid]
        fam_front = pareto_front(fam_cands, obj_list, degradation_key=degradation_key)
        tagged = tag_profiles(
            fam_cands, fam_front, objectives=obj_list, degradation_key=degradation_key,
        )
        per_family[fid] = {
            "family_label": families_data[fid].get("family_label", fid),
            "n_feasible": len(fam_cands),
            "n_pareto": len(fam_front),
            "pareto_front": [c.to_dict() for c in fam_front],
            "tagged": tagged.to_dict(),
        }

    return ParetoAnalysisResult(
        objectives=obj_list,
        degradation_key=degradation_key,
        degradation_label=degradation_label,
        n_feasible_total=len(all_feasible),
        n_pareto_global=len(global_front),
        pareto_front=global_front,
        tagged_global=tag_profiles(
            all_feasible, global_front, objectives=obj_list, degradation_key=degradation_key,
        ),
        per_family=per_family,
    )


def analyze_results_payload(data: Mapping) -> ParetoAnalysisResult:
    return analyze_family_results(
        data.get("families") or {},
        constraints=data.get("constraints"),
    )
