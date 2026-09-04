"""Session evaluation helpers: start state, simulate + score."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from aacopt.capacity import OcvCurve
from aacopt.config import (
    OptimizationSpec, Paths, RewardSpec, SessionSpec, load_config, stage_dir, read_json,
)
from aacopt.degradation import DegradationParameters, HybridDegradationModel
from aacopt.profiles import CCCVFamily, ProfileParams, bind_search, get_family
from aacopt.reward import score_session
from aacopt.simulator import ChargingSimulator


def load_ocv(cell: str) -> OcvCurve:
    npz = stage_dir("01_calibration_dataset", create=False) / "ocv_curves.npz"
    if not npz.is_file():
        raise FileNotFoundError(f"missing {npz} — run scripts/01_build_calibration_dataset.py")
    return OcvCurve.from_npz(npz, cell)


def start_state_and_vnom(cell: str, session: SessionSpec) -> Tuple[Dict[str, float], float]:
    ocv = load_ocv(cell)
    v0 = float(ocv.voltage_from_soc(session.soc_start))
    v_nom = float(ocv.nominal_voltage()) if ocv.nominal_voltage() > 0 else session.v_nom_fallback
    return {
        "soc": float(session.soc_start),
        "v0": v0,
        "t0": float(session.ambient_t0_c),
        "age": 0.0,
        "prev_i": 0.0,
    }, v_nom


def load_degradation_model() -> HybridDegradationModel:
    return HybridDegradationModel(DegradationParameters.load())


def load_anchors(cell: str) -> Dict[str, float]:
    """Optional CCCV 1C bookkeeping from stage 5. Not used in the reward."""
    path = stage_dir("05_reward_anchors", create=False) / "reward_anchors.json"
    if not path.is_file():
        return {}
    blob = read_json(path)
    row = (blob.get("per_cell") or {}).get(cell.upper()) or {}
    return {k: float(row[k]) for k in ("Q_ref", "t_ref_h", "E_ref") if k in row}


def evaluate_params(
    simulator: ChargingSimulator,
    initial_state: Dict[str, float],
    params: ProfileParams,
    *,
    model: HybridDegradationModel,
    spec: RewardSpec,
    anchors: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict, Dict]:
    family = get_family(params.family_id)
    session = simulator.simulate(initial_state, params, family=family)
    metrics = score_session(session, model=model, spec=spec, anchors=anchors)
    return float(metrics["loss"]), metrics, session


def make_simulator(cell: str, device: str = "auto") -> Tuple[ChargingSimulator, Dict[str, float]]:
    bind_search()
    spec = OptimizationSpec.load()
    paths = Paths.load()
    from aacopt.bdt import FrozenBDT
    state, v_nom = start_state_and_vnom(cell, spec.session)
    q_ah = float(load_config("degradation")["physical_constants"]["q_nominal_ah"])
    bdt = FrozenBDT(paths.checkpoint(cell), device=device)
    sim = ChargingSimulator(bdt, q_rated_as=q_ah * 3600.0, session=spec.session, v_nom=v_nom)
    return sim, state


def cccv_params(c_rate: float, *, v_cv: float = 4.20, i_cutoff: float = 0.05) -> ProfileParams:
    q_ah = float(load_config("degradation")["physical_constants"]["q_nominal_ah"])
    bind_search()
    return CCCVFamily.from_dict({
        "i_cc": float(c_rate) * q_ah,
        "v_cv": float(v_cv),
        "i_cutoff": float(i_cutoff),
    })


def jsonable_session(session: Dict) -> Dict:
    out = {}
    for k, v in session.items():
        if hasattr(v, "tolist"):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def jsonable_family_result(result: Dict, *, include_session: bool = True) -> Dict:
    skip = set() if include_session else {"best_session"}
    out = {}
    for k, v in result.items():
        if k in skip:
            continue
        if k == "best_session" and v is not None:
            out[k] = jsonable_session(v)
        else:
            out[k] = v
    return out
