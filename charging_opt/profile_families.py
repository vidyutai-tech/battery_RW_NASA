"""Profile family labels and candidate catalogs for charging optimization."""

from __future__ import annotations

from typing import Dict, List, Tuple

from charging_opt.charging_profile_family import (
    DEFAULT_FAMILY_IDS,
    FAMILY_LABELS as NEW_FAMILY_LABELS,
    ProfileParams,
    get_family,
)
from charging_opt.profile_simulator import ProfileSimulator, ProfileSpec

MIN_REST_MIN = 5.0

# Legacy labels + new family registry
FAMILY_LABELS = {
    "constant_cc": "Constant CC",
    "cc_taper": "CC-taper (2-level)",
    "multi_step_taper": "Multi-step taper",
    "pulsed": "Pulsed charge/rest",
    "other": "Other",
    **NEW_FAMILY_LABELS,
}


def charge_levels(session: Dict) -> List[float]:
    """Distinct positive charge currents (A) in merged segments."""
    segs = ProfileSimulator.merged_segments(session)
    levels = sorted(
        {round(-s["current_a"], 3) for s in segs if s["current_a"] < -1e-6},
    )
    return levels


def profile_family(session: Dict) -> str:
    """Classify a simulated session."""
    if session.get("family_id"):
        return str(session["family_id"])

    spec = session.get("profile_spec") or {}
    if spec.get("family_id"):
        return str(spec["family_id"])

    if float(spec.get("pulse_rest_min", 0.0)) >= MIN_REST_MIN:
        return "pulsed"

    levels = charge_levels(session)
    tapered = any(d.get("ceiling_hit") for d in session.get("decisions", []))
    i_cc = float(spec.get("i_charge", 0.0))
    i_fl = float(spec.get("i_floor", 0.0))

    if len(levels) >= 3 or (tapered and len(levels) >= 3):
        return "multi_step_taper"
    if len(levels) == 2 or (tapered and abs(i_cc - i_fl) > 0.05):
        return "cc_taper"
    if len(levels) <= 1 and abs(i_cc - i_fl) < 0.05 and not tapered:
        return "constant_cc"
    return "other"


def family_from_spec_only(spec: Dict) -> str:
    """Heuristic when full session is unavailable (BO history)."""
    if spec.get("family_id"):
        return str(spec["family_id"])
    if float(spec.get("pulse_rest_min", 0.0)) >= MIN_REST_MIN:
        return "pulsed"
    if "i_charge" in spec and "i_floor" in spec:
        i_cc = float(spec["i_charge"])
        i_fl = float(spec["i_floor"])
        if abs(i_cc - i_fl) < 0.05:
            return "constant_cc"
        if i_cc >= 2.75 and i_fl <= 1.0:
            return "multi_step_taper"
        return "cc_taper"
    return "other"


def simulate_from_spec_dict(sim: ProfileSimulator, start: Dict, spec: Dict) -> Dict:
    """Re-simulate from a stored spec dict (legacy or family-tagged)."""
    if spec.get("family_id"):
        params = ProfileParams.from_dict(spec)
        return sim.simulate_params(start, params)
    return sim.simulate(start, ProfileSpec.from_dict(spec))


def default_candidate_specs() -> List[Tuple[str, str, ProfileSpec]]:
    """Legacy CC-taper candidates for post-hoc family comparison."""
    return [
        ("const_0.75A", "constant_cc", ProfileSpec.cc_taper(0.75, 0.75)),
        ("const_1.0A", "constant_cc", ProfileSpec.cc_taper(1.0, 0.75)),
        ("taper_1.25A", "cc_taper", ProfileSpec.cc_taper(1.25, 0.75)),
        ("taper_2.0A", "cc_taper", ProfileSpec.cc_taper(2.0, 0.75)),
        ("multistep_3.0A", "multi_step_taper", ProfileSpec.cc_taper(3.0, 0.75)),
        ("multistep_3.5A", "multi_step_taper", ProfileSpec.cc_taper(3.5, 0.75)),
        ("multistep_4.0A", "multi_step_taper", ProfileSpec.cc_taper(4.0, 0.75)),
        ("pulsed_2A_10on_5rest", "pulsed", ProfileSpec(2.0, 10.0, 5.0, 0.75)),
        ("pulsed_1.5A_15on_10rest", "pulsed", ProfileSpec(1.5, 15.0, 10.0, 0.75)),
        ("pulsed_2.5A_20on_5rest", "pulsed", ProfileSpec(2.5, 20.0, 5.0, 0.75)),
    ]


__all__ = [
    "DEFAULT_FAMILY_IDS",
    "FAMILY_LABELS",
    "charge_levels",
    "profile_family",
    "family_from_spec_only",
    "simulate_from_spec_dict",
    "default_candidate_specs",
]
