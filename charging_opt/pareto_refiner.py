"""
Second-stage Pareto front densification — focus BO on sparse duration gaps.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from charging_opt.pareto_analysis import ParetoCandidate


def find_sparse_duration_gaps(
    front: Sequence[ParetoCandidate],
    *,
    n_targets: int = 5,
    duration_min: float = 50.0,
    duration_max: float = 105.0,
) -> List[float]:
    """Return midpoints of the largest gaps on the duration axis."""
    if len(front) < 2:
        step = (duration_max - duration_min) / max(n_targets, 1)
        return [duration_min + step * (i + 0.5) for i in range(n_targets)]

    durations = sorted(c.duration_min for c in front)
    gaps = [
        (durations[i + 1] - durations[i], (durations[i] + durations[i + 1]) / 2.0)
        for i in range(len(durations) - 1)
    ]
    gaps.sort(key=lambda g: g[0], reverse=True)
    return [mid for _, mid in gaps[:n_targets]]


def duration_constrained_loss(
    base_loss: float,
    duration_min: float,
    *,
    target_duration: float,
    tolerance_min: float = 3.0,
    penalty_weight: float = 50.0,
) -> float:
    """Soft penalty pushing the optimizer toward a target charge duration."""
    deviation = max(0.0, abs(duration_min - target_duration) - tolerance_min)
    if tolerance_min <= 0:
        return base_loss
    return float(base_loss + penalty_weight * (deviation / tolerance_min) ** 2)


def refinement_targets_from_results(
    results_payload: Dict,
    *,
    n_targets: int = 5,
) -> List[Dict]:
    """
    Build refinement jobs from family_optimization_results JSON payload.

    Returns list of {target_duration_min, family_id, gap_rank}.
    """
    from charging_opt.pareto_analysis import analyze_family_results

    families = results_payload.get("families", results_payload)
    analysis = analyze_family_results(families)
    gaps = find_sparse_duration_gaps(analysis.pareto_front, n_targets=n_targets)
    jobs = []
    for rank, target in enumerate(gaps):
        jobs.append({
            "target_duration_min": float(target),
            "gap_rank": rank,
            "family_id": "pulsed",
        })
    return jobs
