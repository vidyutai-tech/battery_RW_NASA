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

    def objective_vector(self) -> List[float]:
        return [
            self.duration_min,
            self.sei_per_pct_soc,
            self.voltage_stress_v2_min,
            self.temperature_penalty_c2_min,
        ]

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
    n_feasible_total: int = 0
    n_pareto_global: int = 0
    pareto_front: List[ParetoCandidate] = field(default_factory=list)
    tagged_global: TaggedProfiles = field(default_factory=TaggedProfiles)
    per_family: Dict[str, Dict] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "objectives": self.objectives,
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
    if sei is None:
        return None
    return ParetoCandidate(
        family_id=family_id,
        family_label=family_label,
        params=dict(entry.get("params") or {}),
        duration_min=_metric_float(metrics, "duration_min"),
        sei_per_pct_soc=float(sei),
        voltage_stress_v2_min=_metric_float(metrics, "voltage_stress_v2_min"),
        temperature_penalty_c2_min=_metric_float(metrics, "temperature_penalty_c2_min"),
        loss=_metric_float(metrics, "loss", _metric_float(entry, "loss")),
        peak_voltage=_metric_float(metrics, "peak_voltage"),
        peak_temperature=_metric_float(metrics, "peak_temperature"),
        end_reason=str(metrics.get("end_reason", entry.get("end_reason", ""))),
    )


def extract_feasible_candidates(families_data: Mapping[str, Mapping]) -> List[ParetoCandidate]:
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
            key = (
                fid,
                round(cand.duration_min, 3),
                round(cand.sei_per_pct_soc, 4),
                round(cand.voltage_stress_v2_min, 4),
                tuple(sorted((k, round(float(v), 6)) for k, v in cand.params.items()
                              if k != "family_id" and isinstance(v, (int, float)))),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(cand)
    return out


def dominates(a: ParetoCandidate, b: ParetoCandidate) -> bool:
    """True if *a* is no worse on all objectives and strictly better on at least one."""
    av = a.objective_vector()
    bv = b.objective_vector()
    if any(x > y for x, y in zip(av, bv)):
        return False
    return any(x < y for x, y in zip(av, bv))


def pareto_front(candidates: Sequence[ParetoCandidate]) -> List[ParetoCandidate]:
    if not candidates:
        return []
    front: List[ParetoCandidate] = []
    for i, a in enumerate(candidates):
        if any(i != j and dominates(b, a) for j, b in enumerate(candidates)):
            continue
        front.append(a)
    return sorted(front, key=lambda c: (c.duration_min, c.sei_per_pct_soc))


def select_fastest(candidates: Sequence[ParetoCandidate]) -> Optional[ParetoCandidate]:
    if not candidates:
        return None
    return min(candidates, key=lambda c: (c.duration_min, c.sei_per_pct_soc))


def select_lifetime(candidates: Sequence[ParetoCandidate]) -> Optional[ParetoCandidate]:
    if not candidates:
        return None
    return min(candidates, key=lambda c: (c.sei_per_pct_soc, c.duration_min))


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

    utopia = {
        "duration_min": min(c.duration_min for c in front),
        "sei_per_pct_soc": min(c.sei_per_pct_soc for c in front),
        "voltage_stress_v2_min": min(c.voltage_stress_v2_min for c in front),
        "temperature_penalty_c2_min": min(c.temperature_penalty_c2_min for c in front),
    }
    span = {
        k: max(getattr(c, k) for c in front) - utopia[k] + 1e-9
        for k in objectives
    }

    def _score(c: ParetoCandidate) -> float:
        return sum(
            ((getattr(c, k) - utopia[k]) / span[k]) ** 2
            for k in objectives
        )

    return min(front, key=_score)


def tag_profiles(
    all_feasible: Sequence[ParetoCandidate],
    front: Sequence[ParetoCandidate],
) -> TaggedProfiles:
    """
    Fastest / Lifetime from all feasible candidates; Balanced from the Pareto front.
    """
    return TaggedProfiles(
        fastest=select_fastest(all_feasible),
        lifetime=select_lifetime(all_feasible),
        balanced=select_balanced(front),
    )


def analyze_family_results(families_data: Mapping[str, Mapping]) -> ParetoAnalysisResult:
    all_feasible = extract_feasible_candidates(families_data)
    global_front = pareto_front(all_feasible)

    per_family: Dict[str, Dict] = {}
    for fid in families_data:
        fam_cands = [c for c in all_feasible if c.family_id == fid]
        fam_front = pareto_front(fam_cands)
        tagged = tag_profiles(fam_cands, fam_front)
        per_family[fid] = {
            "family_label": families_data[fid].get("family_label", fid),
            "n_feasible": len(fam_cands),
            "n_pareto": len(fam_front),
            "pareto_front": [c.to_dict() for c in fam_front],
            "tagged": tagged.to_dict(),
        }

    return ParetoAnalysisResult(
        n_feasible_total=len(all_feasible),
        n_pareto_global=len(global_front),
        pareto_front=global_front,
        tagged_global=tag_profiles(all_feasible, global_front),
        per_family=per_family,
    )


def analyze_results_payload(data: Mapping) -> ParetoAnalysisResult:
    return analyze_family_results(data.get("families") or {})
