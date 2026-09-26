"""Shared helpers for Final_Aging_based_setup. Call existing aacopt APIs only."""

from __future__ import annotations

import csv
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

SCRIPTS_FINAL = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPTS_FINAL.parents[1]
REPO_ROOT = PROJECT_DIR.parent
SRC = PROJECT_DIR / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import aacopt.profiles as profiles  # noqa: E402
from aacopt.bdt import FrozenBDT  # noqa: E402
from aacopt.config import (  # noqa: E402
    OptimizationSpec, Paths, RESULTS_DIR, RewardSpec, file_hash, load_config,
    provenance, read_json, write_json,
)
from aacopt.evaluate import (  # noqa: E402
    cccv_params, evaluate_params, jsonable_family_result, load_anchors,
    load_degradation_model, start_state_and_vnom,
)
from aacopt.optimizers.gp_bo import (  # noqa: E402
    DUPLICATE_JITTER_SCALE, _vector_key, ensure_twin_eval, optimize_family_gp_bo,
)
from aacopt.profiles import get_family  # noqa: E402
from aacopt.simulator import ChargingSimulator  # noqa: E402
from aacopt.viz.style import FAMILY_LABELS, FAMILY_ORDER  # noqa: E402

CELLS = ("RW9", "RW10", "RW11", "RW12")
N_ELITE = 6
N_GRID = 100
FINE_TUNE_FRACTION = 0.60
AUTHOR_SPLIT_SEED = 42
AUTHOR_TRAIN_FRAC = 0.6
AUTHOR_VAL_FRAC = 0.2
CHUNK_SIZE = 150
OUTPUT_ROOT = RESULTS_DIR / "Final_Aging_based_setup"
SOURCE_PT_REL = Path("outputs/twin_source/20260610_111409/twin_source_RW9.pt")
CELL_COLORS = {"RW9": "#2563eb", "RW10": "#16a34a", "RW11": "#ea580c", "RW12": "#7c3aed"}

LossTransform = Callable[[float, Dict[str, Any], Dict[str, Any]], Tuple[float, Dict[str, Any]]]


def output_root(*, create: bool = True) -> Path:
    if create:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    return OUTPUT_ROOT


def apply_i_max(i_max_a: float) -> None:
    """Keep session/families from YAML; replace only the current upper bound."""
    spec = OptimizationSpec.load()
    profiles.bind_search(spec)
    profiles._space = replace(spec.space, i_max_a=float(i_max_a))


def make_simulator_ckpt(
    cell: str,
    *,
    device: str = "auto",
    i_max_a: float,
    checkpoint: Optional[Path] = None,
) -> Tuple[ChargingSimulator, Dict[str, float]]:
    """Like evaluate.make_simulator, but does not reset i_max from YAML."""
    spec = OptimizationSpec.load()
    paths = Paths.load()
    apply_i_max(i_max_a)
    state, v_nom = start_state_and_vnom(cell, spec.session)
    q_ah = float(load_config("degradation")["physical_constants"]["q_nominal_ah"])
    ckpt = Path(checkpoint) if checkpoint is not None else paths.checkpoint(cell)
    bdt = FrozenBDT(ckpt, device=device)
    sim = ChargingSimulator(bdt, q_rated_as=q_ah * 3600.0, session=spec.session, v_nom=v_nom)
    apply_i_max(i_max_a)
    return sim, state


def author_splits(cell: str):
    from rw_transfer.data.author_dataset import AuthorChunkDataset, random_split_author_dataset
    from rw_transfer.data.author_loader import load_author_stitched_series

    paths = Paths.load()
    stitched = load_author_stitched_series(str(paths.matlab_dir), cell, decimation=1)
    dataset = AuthorChunkDataset(stitched, chunk_size=CHUNK_SIZE)
    train, val, test = random_split_author_dataset(
        dataset, train_frac=AUTHOR_TRAIN_FRAC, val_frac=AUTHOR_VAL_FRAC, seed=AUTHOR_SPLIT_SEED,
    )
    return stitched, dataset, train, val, test


def training_subset(cell: str, *, source_twin: bool = False):
    """Training windows actually used by that cell's inherited twin."""
    from rw_transfer.data.author_dataset import subset_author_train_by_fraction

    _, _, train, _, _ = author_splits(cell if not source_twin else "RW9")
    if source_twin or cell.upper() == "RW9":
        return train, "author_random_60_20_20_seed42_train"
    adapted = subset_author_train_by_fraction(train, FINE_TUNE_FRACTION)
    return adapted, f"author_train_prefix_frac{FINE_TUNE_FRACTION:.2f}"


def params_from_dict(raw: Dict[str, Any]):
    blob = dict(raw)
    fid = str(blob.pop("family_id"))
    cls = type(get_family(fid))
    return cls.from_dict(blob)


def charge_i(session: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    t = np.asarray(session["time_s"], dtype=np.float64)
    i = -np.asarray(session["current_a"], dtype=np.float64)
    return t, i


def resample_unit_time(t: np.ndarray, y: np.ndarray, n: int = N_GRID) -> np.ndarray:
    if t.size == 0:
        return np.zeros(n, dtype=np.float64)
    t = t - t[0]
    span = float(t[-1]) if t.size else 0.0
    if span <= 1e-9:
        return np.full(n, float(y[0]) if y.size else 0.0, dtype=np.float64)
    tau = t / span
    grid = np.linspace(0.0, 1.0, n)
    return np.interp(grid, tau, y)


def mean_std_cv(xs: List[float]) -> Dict[str, float]:
    a = np.asarray(xs, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "cv_pct": float("nan"), "min": float("nan"), "max": float("nan")}
    mu = float(a.mean())
    sd = float(a.std(ddof=1)) if a.size > 1 else 0.0
    cv = 100.0 * sd / abs(mu) if abs(mu) > 1e-18 else float("nan")
    return {"n": int(a.size), "mean": mu, "std": sd, "cv_pct": cv,
            "min": float(a.min()), "max": float(a.max())}


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def load_elite(gpbo_dir: Path, cells: Sequence[str], k: int) -> List[Dict[str, Any]]:
    elite: List[Dict[str, Any]] = []
    for cell in cells:
        rows: List[Dict[str, Any]] = []
        for fid in FAMILY_ORDER:
            path = gpbo_dir / cell / f"{fid}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            blob = read_json(path)
            for idx, h in enumerate(blob.get("history") or []):
                m = h.get("metrics") or {}
                if not bool(m.get("feasible", h.get("feasible"))):
                    continue
                rows.append({
                    "source_cell": cell,
                    "family_id": str((h.get("params") or {}).get("family_id", fid)),
                    "history_index": int(idx),
                    "loss": float(h.get("loss", m.get("loss", np.inf))),
                    "reward": float(h.get("reward", m.get("reward", np.nan))),
                    "params": dict(h.get("params") or {}),
                    "native_metrics": {kk: vv for kk, vv in m.items()
                                       if kk not in ("weights", "anchors")},
                })
        rows.sort(key=lambda r: r["loss"])
        for rank, row in enumerate(rows[:k], start=1):
            row["rank"] = rank
            row["profile_id"] = f"{cell}-P{rank}"
            elite.append(row)
    return elite


def evaluate_criteria(
    *,
    family_counts: Dict[str, Counter],
    stats: Dict[str, Dict[str, float]],
    elite: List[Dict[str, Any]],
    cross: List[Dict[str, Any]],
    cells: Sequence[str] = CELLS,
    k: int = N_ELITE,
) -> Dict[str, Any]:
    per_cell_c1 = {}
    c1_ok = True
    for cell in cells:
        counts = family_counts[cell]
        if not counts:
            per_cell_c1[cell] = {"dominant_family": None, "count": 0, "pass": False}
            c1_ok = False
            continue
        dominant, n = counts.most_common(1)[0]
        passed = int(n) >= 4
        per_cell_c1[cell] = {
            "dominant_family": dominant,
            "count": int(n),
            "pass": passed,
            "counts": dict(counts),
        }
        if not passed:
            c1_ok = False

    i_max_stats = stats["I_max (A)"]
    cv = float(i_max_stats["cv_pct"])
    c2_ok = bool(np.isfinite(cv) and cv < 15.0)

    per_cell_c3 = {}
    c3_ok = True
    for cell in cells:
        pids = [e["profile_id"] for e in elite if e["source_cell"] == cell]
        n_all = 0
        detail = []
        for pid in pids:
            hits = [r for r in cross if r["profile_id"] == pid]
            all_ok = bool(hits) and all(bool(r["feasible"]) for r in hits)
            n_all += int(all_ok)
            detail.append({"profile_id": pid, "feasible_on_all_twins": all_ok})
        passed = n_all >= 4
        per_cell_c3[cell] = {"n_feasible_on_all_twins": n_all, "of": k, "pass": passed,
                             "profiles": detail}
        if not passed:
            c3_ok = False

    verdict = "PASS" if (c1_ok and c2_ok and c3_ok) else "FAIL"
    return {
        "C1": {"pass": c1_ok, "per_cell": per_cell_c1,
               "rule": "For every cell, the same dominant family appears in >= 4 of its Top-6."},
        "C2": {"pass": c2_ok, "cv_pct_peak_current": cv, "threshold_pct": 15.0,
               "n_elites": int(i_max_stats["n"]),
               "mean": i_max_stats["mean"], "std": i_max_stats["std"],
               "min": i_max_stats["min"], "max": i_max_stats["max"],
               "rule": "Coefficient of variation of peak current across all 24 elites < 15%."},
        "C3": {"pass": c3_ok, "per_cell": per_cell_c3,
               "rule": "For every cell, >= 4 of its 6 elites are feasible (V_peak <= 4.20 V and energy target met) on ALL four twins."},
        "verdict": verdict,
        "all_hold": verdict == "PASS",
    }


def optimize_family_gp_bo_adapted(
    simulator: ChargingSimulator,
    initial_state: Dict[str, float],
    family_id: str,
    *,
    model,
    spec,
    anchors: Dict[str, float],
    n_calls: int,
    n_initial_points: int,
    seed: int,
    acq_func: str = "EI",
    acq_xi: float = 0.01,
    noise: float = 1e-6,
    loss_transform: Optional[LossTransform] = None,
) -> Dict[str, Any]:
    """Copy of optimize_family_gp_bo with an optional post-score loss transform (Phase 3)."""
    from skopt import gp_minimize

    ensure_twin_eval(simulator)
    family = get_family(family_id)
    cls = type(family)
    history: List[Dict[str, Any]] = []
    seen = set()
    rng = np.random.default_rng(int(seed))
    space = cls.search_space()
    lo = [float(d.low) for d in space]
    hi = [float(d.high) for d in space]

    def _jitter(x: List[float]) -> List[float]:
        key = _vector_key(x)
        if key not in seen:
            return list(x)
        out = list(x)
        for _ in range(8):
            for i, (a, b) in enumerate(zip(lo, hi)):
                span = max(b - a, 1e-9)
                out[i] = float(np.clip(out[i] + rng.normal(0.0, DUPLICATE_JITTER_SCALE * span), a, b))
            if _vector_key(out) not in seen:
                return out
        return out

    def objective(x: List[float]) -> float:
        ensure_twin_eval(simulator)
        x = _jitter(list(x))
        seen.add(_vector_key(x))
        params = cls.from_vector(x)
        loss, metrics, session = evaluate_params(
            simulator, initial_state, params, model=model, spec=spec, anchors=anchors,
        )
        if loss_transform is not None:
            loss, metrics = loss_transform(float(loss), metrics, session)
        history.append({
            "family_id": family_id,
            "params": params.to_dict(),
            "loss": float(loss),
            "reward": float(metrics["reward"]),
            "feasible": bool(metrics["feasible"]),
            "metrics": {k: v for k, v in metrics.items() if k not in ("weights", "anchors")},
        })
        return float(loss)

    seeds = list(cls.seed_points())
    n_seed = len(seeds)
    n_random_extra = max(0, int(n_initial_points) - n_seed)
    n_calls_eff = max(int(n_calls), n_seed + n_random_extra)
    gp_kwargs: Dict[str, Any] = {
        "func": objective,
        "dimensions": space,
        "n_calls": n_calls_eff,
        "n_initial_points": n_random_extra,
        "x0": seeds if seeds else None,
        "random_state": int(seed),
        "acq_func": acq_func,
        "noise": float(noise),
    }
    if acq_func == "EI":
        gp_kwargs["xi"] = float(acq_xi)
    skopt_result = gp_minimize(**gp_kwargs)

    feasible = [h for h in history if h["feasible"]]
    pool = feasible if feasible else history
    best = min(pool, key=lambda h: h["loss"])
    best_params = cls.from_dict(best["params"])
    ensure_twin_eval(simulator)
    _, best_metrics, best_session = evaluate_params(
        simulator, initial_state, best_params, model=model, spec=spec, anchors=anchors,
    )
    if loss_transform is not None:
        best_loss, best_metrics = loss_transform(float(best_metrics["loss"]), best_metrics, best_session)
    else:
        best_loss = float(best["loss"])
    return {
        "family_id": family_id,
        "family_label": family.label,
        "method": "gp_bo",
        "best_params": best_params.to_dict(),
        "best_loss": float(best_loss),
        "best_reward": float(best_metrics["reward"]),
        "best_metrics": best_metrics,
        "best_session": best_session,
        "history": history,
        "n_evaluated": len(history),
        "n_calls": n_calls_eff,
        "n_seed_points": n_seed,
        "n_initial_points": int(n_initial_points),
        "acq_func": acq_func,
        "seed": int(seed),
        "skopt_fun": float(skopt_result.fun) if skopt_result is not None else None,
        "warm_start_from_random_search": False,
    }


def make_rescale_transform(
    baseline: Dict[str, Any],
    *,
    drop_soc: bool = False,
) -> LossTransform:
    """Normalize each published reward term by the cell's 0.5C CCCV baseline term."""
    reward_spec = RewardSpec.load()
    base_soc = abs(float(baseline["soc_reward"]))
    base_q = abs(float(baseline["qloss_penalty"]))
    base_t = abs(float(baseline["time_penalty"]))
    if base_soc < 1e-18 or base_q < 1e-18 or base_t < 1e-18:
        raise RuntimeError(f"0.5C CCCV baseline term is zero; cannot rescale: {baseline}")

    def transform(loss: float, metrics: Dict[str, Any], session: Dict[str, Any]):
        del loss, session
        soc_n = 0.0 if drop_soc else float(metrics["soc_reward"]) / base_soc
        q_n = float(metrics["qloss_penalty"]) / base_q
        t_n = float(metrics["time_penalty"]) / base_t
        reward = float(soc_n - q_n - t_n)
        if not np.isfinite(reward):
            reward = -1e6
        duration_min = float(metrics["duration_min"])
        penalty = float(metrics["penalty"])
        new_loss = (
            -reward
            + float(reward_spec.duration_loss_weight) * duration_min
            + penalty
        )
        out = dict(metrics)
        out["reward_unnormalized"] = float(metrics["reward"])
        out["loss_unnormalized"] = float(metrics["loss"])
        out["soc_reward"] = float(soc_n)
        out["qloss_penalty"] = float(q_n)
        out["time_penalty"] = float(t_n)
        out["reward"] = float(reward)
        out["loss"] = float(new_loss)
        out["rescale_baseline_soc_reward"] = float(baseline["soc_reward"])
        out["rescale_baseline_qloss_penalty"] = float(baseline["qloss_penalty"])
        out["rescale_baseline_time_penalty"] = float(baseline["time_penalty"])
        out["drop_soc_term"] = bool(drop_soc)
        return float(new_loss), out

    return transform


def run_gp_bo_cell(
    cell: str,
    out_dir: Path,
    *,
    device: str = "auto",
    i_max_a: float,
    checkpoint: Optional[Path] = None,
    families: Optional[List[str]] = None,
    resume: bool = True,
    seed_cell: Optional[str] = None,
    loss_transform: Optional[LossTransform] = None,
    include_session: bool = True,
) -> Dict[str, Any]:
    import time

    apply_i_max(i_max_a)
    opt = OptimizationSpec.load()
    reward = RewardSpec.load()
    paths = Paths.load()
    model = load_degradation_model()
    anchors = load_anchors(cell)
    simulator, state = make_simulator_ckpt(
        cell, device=device, i_max_a=i_max_a, checkpoint=checkpoint,
    )
    fams = list(families or opt.families)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_cell = (seed_cell or cell).upper()
    ckpt_path = Path(checkpoint) if checkpoint is not None else paths.checkpoint(cell)

    family_results: Dict[str, Any] = {}
    t0 = time.perf_counter()
    for fid in fams:
        fam_path = out_dir / f"{fid}.json"
        if resume and fam_path.is_file():
            print(f"  [{cell} gp_bo] skip existing {fid}", flush=True)
            family_results[fid] = read_json(fam_path)
            continue
        seed = opt.seed_for(seed_cell, fid, "gp_bo")
        print(f"  [{cell} gp_bo] {fid}  n={opt.n_evals_per_family}  seed={seed}  i_max={i_max_a}", flush=True)
        ft = time.perf_counter()
        apply_i_max(i_max_a)
        kwargs = dict(
            simulator=simulator, initial_state=state, family_id=fid,
            model=model, spec=reward, anchors=anchors,
            n_calls=opt.n_evals_per_family,
            n_initial_points=int(opt.gp_bo["n_initial_points"]),
            seed=seed,
            acq_func=str(opt.gp_bo.get("acq_func", "EI")),
            acq_xi=float(opt.gp_bo.get("acq_xi", 0.01)),
            noise=float(opt.gp_bo.get("noise", 1e-6)),
        )
        if loss_transform is None:
            result = optimize_family_gp_bo(**kwargs)
        else:
            result = optimize_family_gp_bo_adapted(**kwargs, loss_transform=loss_transform)
        result["wall_clock_s"] = float(time.perf_counter() - ft)
        payload = jsonable_family_result(result, include_session=include_session)
        write_json(fam_path, payload)
        family_results[fid] = payload
        m = result["best_metrics"]
        print(
            f"    best R={m['reward']:.6g}  loss={result['best_loss']:.6g}  "
            f"feas={m['feasible']}  t={m['duration_min']:.4g} min  "
            f"Q={m['q_total']:.6g}  ΔSOC={m['delta_soc']:.6g}  [{m['end_reason']}]",
            flush=True,
        )

    from aacopt.opt_driver import evaluate_baselines
    apply_i_max(i_max_a)
    baselines = evaluate_baselines(simulator, state, model=model, spec=reward, anchors=anchors)
    write_json(out_dir / "baselines.json", baselines)

    feasible_any = any(
        (family_results[f].get("best_metrics") or {}).get("feasible") for f in fams
        if f in family_results
    )
    summary = {
        "cell": cell.upper(),
        "method": "gp_bo",
        "families": fams,
        "n_evals_per_family": opt.n_evals_per_family,
        "device": str(simulator.bdt.device),
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": file_hash(ckpt_path),
        "start_state": state,
        "v_nom": simulator.v_nom,
        "anchors": anchors,
        "reward": reward.to_dict(),
        "i_max_a": float(i_max_a),
        "seed_cell": seed_cell,
        "seeds": {fid: opt.seed_for(seed_cell, fid, "gp_bo") for fid in fams},
        "loss_transform": None if loss_transform is None else "phase3_rescale",
        "baselines": baselines,
        "best_by_family": {
            fid: {
                "best_params": family_results[fid]["best_params"],
                "best_loss": family_results[fid]["best_loss"],
                "best_reward": family_results[fid]["best_reward"],
                "feasible": family_results[fid]["best_metrics"]["feasible"],
                "q_total": family_results[fid]["best_metrics"]["q_total"],
                "duration_min": family_results[fid]["best_metrics"].get(
                    "duration_min", family_results[fid]["best_metrics"]["duration_h"] * 60.0
                ),
                "peak_t": family_results[fid]["best_metrics"].get("peak_t"),
                "end_reason": family_results[fid]["best_metrics"]["end_reason"],
            }
            for fid in fams if fid in family_results
        },
        "any_feasible": bool(feasible_any),
        "wall_clock_s": float(time.perf_counter() - t0),
        "provenance": provenance(
            "Final_Aging_based_setup",
            configs=["paths", "degradation_fitted", "reward", "optimization"],
            inputs=[ckpt_path],
        ),
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def _flat_params(p: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("i_cc", "v_cv", "i_cutoff", "i1", "i2", "i3", "soc_switch", "soc1", "soc2",
            "i_charge", "pulse_on_min", "pulse_rest_min", "rest_fraction", "i_floor")
    return {k: p.get(k) for k in keys}


def _param_short(p: Dict[str, Any]) -> str:
    fid = p.get("family_id")
    if fid == "cccv":
        return f"I={p['i_cc']:.6g}A Vcv={p['v_cv']:.6g}"
    if fid == "two_step":
        return f"I1={p['i1']:.6g} I2={p['i2']:.6g}"
    if fid == "three_step":
        return f"I1={p['i1']:.6g} I2={p['i2']:.6g} I3={p['i3']:.6g}"
    if fid == "pulsed":
        return f"Ion={p['i_charge']:.6g} on={p['pulse_on_min']:.6g} rest={p['pulse_rest_min']:.6g}"
    return ""


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" if i == 0 else "---:" for i in range(len(headers))) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join([line, sep, body])


def run_litmus(
    gpbo_dir: Path,
    out: Path,
    *,
    device: str = "auto",
    i_max_a: float,
    cells: Sequence[str] = CELLS,
    k: int = N_ELITE,
    eval_cells: Optional[Sequence[str]] = None,
    checkpoints: Optional[Dict[str, Path]] = None,
) -> Dict[str, Any]:
    """Replay elites on twins and score C1–C3. Writes under ``out`` only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from aacopt.config import ensure_matplotlib_cache
    from aacopt.viz.style import apply_paper_style, savefig

    ensure_matplotlib_cache()
    opt = OptimizationSpec.load()
    reward = RewardSpec.load()
    paths = Paths.load()
    eval_cells = list(eval_cells or cells)
    out = Path(out)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    apply_i_max(i_max_a)
    elite = load_elite(gpbo_dir, list(cells), k)
    family_counts = {c: Counter() for c in cells}
    for e in elite:
        family_counts[e["source_cell"]][e["family_id"]] += 1

    model = load_degradation_model()
    sims = {}
    states = {}
    anchors = {}
    for cell in eval_cells:
        print(f"Loading twin {cell} …", flush=True)
        ckpt = (checkpoints or {}).get(cell)
        sims[cell], states[cell] = make_simulator_ckpt(
            cell if cell in CELLS else "RW9",
            device=device, i_max_a=i_max_a, checkpoint=ckpt,
        )
        anchors[cell] = load_anchors(cell if cell in CELLS else "RW9")

    cross_rows: List[Dict[str, Any]] = []
    native: List[Dict[str, Any]] = []
    native_I: List[np.ndarray] = []
    ids: List[str] = []

    n_eval = 0
    total = len(elite) * len(eval_cells)
    for e in elite:
        params = params_from_dict(e["params"])
        e["params"] = params.to_dict()
        for cell in eval_cells:
            n_eval += 1
            print(f"  [{n_eval}/{total}] {e['profile_id']} → {cell}", flush=True)
            apply_i_max(i_max_a)
            loss, metrics, session = evaluate_params(
                sims[cell], states[cell], params, model=model, spec=reward,
                anchors=anchors[cell],
            )
            t, i = charge_i(session)
            rec = {
                "profile_id": e["profile_id"],
                "source_cell": e["source_cell"],
                "rank": e["rank"],
                "family_id": e["family_id"],
                "eval_cell": cell,
                "native": cell == e["source_cell"],
                "feasible": bool(metrics["feasible"]),
                "loss": float(loss),
                "reward": float(metrics["reward"]),
                "duration_min": float(metrics["duration_min"]),
                "q_total": float(metrics["q_total"]),
                "peak_v": float(metrics["peak_v"]),
                "peak_t": float(metrics["peak_t"]),
                "delta_soc": float(metrics["delta_soc"]),
                "soc_reward": float(metrics["soc_reward"]),
                "qloss_penalty": float(metrics["qloss_penalty"]),
                "time_penalty": float(metrics["time_penalty"]),
                "i_mean": float(i.mean()) if i.size else float("nan"),
                "i_max": float(i.max()) if i.size else float("nan"),
                "end_reason": metrics.get("end_reason"),
            }
            rec.update({f"p_{kk}": vv for kk, vv in _flat_params(e["params"]).items()})
            cross_rows.append(rec)
            if cell == e["source_cell"]:
                e["native_replay"] = rec
                native.append({
                    "profile_id": e["profile_id"],
                    "source_cell": e["source_cell"],
                    "rank": e["rank"],
                    "family_id": e["family_id"],
                    "t_s": t,
                    "i_a": i,
                })
                native_I.append(resample_unit_time(t, i))
                ids.append(e["profile_id"])

    I = np.stack(native_I, axis=0) if native_I else np.zeros((0, N_GRID))
    dist = np.sqrt(((I[:, None, :] - I[None, :, :]) ** 2).mean(axis=2)) if I.size else np.zeros((0, 0))

    within = {}
    between_vals = []
    n_cells_src = len(list(cells))
    for ci, c in enumerate(cells):
        sl = slice(ci * k, (ci + 1) * k)
        sub = dist[sl, sl] if dist.size else np.zeros((0, 0))
        tri = sub[np.triu_indices(min(k, sub.shape[0]), k=1)] if sub.size else np.array([])
        within[c] = float(tri.mean()) if tri.size else 0.0
    for i in range(len(ids)):
        ci = i // k
        for j in range(i + 1, len(ids)):
            if j // k != ci:
                between_vals.append(float(dist[i, j]))
    between = float(np.mean(between_vals)) if between_vals else float("nan")

    i_mean = [e["native_replay"]["i_mean"] for e in elite]
    i_max = [e["native_replay"]["i_max"] for e in elite]
    t_min = [e["native_replay"]["duration_min"] for e in elite]
    qv = [e["native_replay"]["q_total"] for e in elite]
    stats = {
        "I_mean (A)": mean_std_cv(i_mean),
        "I_max (A)": mean_std_cv(i_max),
        "Time (min)": mean_std_cv(t_min),
        "Q_loss": mean_std_cv(qv),
        "Reward": mean_std_cv([e["native_replay"]["reward"] for e in elite]),
        "V_peak (V)": mean_std_cv([e["native_replay"]["peak_v"] for e in elite]),
    }

    n_all_feasible = 0
    for e in elite:
        hits = [r for r in cross_rows if r["profile_id"] == e["profile_id"]]
        if all(r["feasible"] for r in hits):
            n_all_feasible += 1

    criteria = evaluate_criteria(
        family_counts=family_counts, stats=stats, elite=elite, cross=cross_rows,
        cells=cells, k=k,
    )

    voltage_headroom = {}
    for cell in cells:
        bl_path = gpbo_dir / cell / "baselines.json"
        if not bl_path.is_file():
            continue
        bl = read_json(bl_path)
        by_name = {b["name"]: b for b in bl}
        p1 = float(by_name["CCCV 1C"]["metrics"]["peak_v"])
        feas2 = bool(by_name["CCCV 2C"]["metrics"]["feasible"])
        voltage_headroom[cell] = {
            "peak_v_1c": p1,
            "headroom_1c_mv": 1000.0 * (4.20 - p1),
            "cccv_2c_feasible": feas2,
        }

    feas_grid = np.zeros((len(ids), len(eval_cells)))
    time_grid = np.full((len(ids), len(eval_cells)), np.nan)
    q_grid = np.full((len(ids), len(eval_cells)), np.nan)
    id_index = {pid: i for i, pid in enumerate(ids)}
    cell_index = {c: j for j, c in enumerate(eval_cells)}
    for r in cross_rows:
        i = id_index[r["profile_id"]]
        j = cell_index[r["eval_cell"]]
        feas_grid[i, j] = 1.0 if r["feasible"] else 0.0
        time_grid[i, j] = r["duration_min"]
        q_grid[i, j] = r["q_total"]

    master_fields = [
        "profile_id", "source_cell", "rank", "family_id", "loss", "reward",
        "duration_min", "q_total", "peak_v", "peak_t", "delta_soc",
        "soc_reward", "qloss_penalty", "time_penalty",
        "i_mean", "i_max",
        "p_i_cc", "p_v_cv", "p_i_cutoff", "p_i1", "p_i2", "p_i3", "p_soc_switch",
        "p_soc1", "p_soc2", "p_i_charge", "p_pulse_on_min", "p_pulse_rest_min",
        "p_rest_fraction", "p_i_floor",
    ]
    master_rows = []
    for e in elite:
        row = dict(e["native_replay"])
        row["loss"] = e["loss"]
        master_rows.append(row)

    write_csv(out / "elite_master.csv", master_rows, master_fields)
    write_csv(out / "cross_cell.csv", cross_rows, [
        "profile_id", "source_cell", "rank", "family_id", "eval_cell", "native",
        "feasible", "loss", "reward", "duration_min", "q_total", "peak_v", "peak_t",
        "delta_soc", "soc_reward", "qloss_penalty", "time_penalty",
        "i_mean", "i_max", "end_reason",
    ])
    count_rows = [{"family": FAMILY_LABELS[fid], **{c: family_counts[c][fid] for c in cells}}
                  for fid in FAMILY_ORDER]
    write_csv(out / "family_counts.csv", count_rows, ["family", *list(cells)])
    stat_rows = [{"quantity": name, **s} for name, s in stats.items()]
    write_csv(out / "parameter_spread.csv", stat_rows,
              ["quantity", "n", "mean", "std", "cv_pct", "min", "max"])
    dist_rows = [{"profile_id": ids[i], **{ids[j]: float(dist[i, j]) for j in range(len(ids))}}
                 for i in range(len(ids))]
    write_csv(out / "similarity_rmse_I.csv", dist_rows, ["profile_id", *ids])

    payload = {
        "k": k,
        "n_elite": len(elite),
        "family_counts": {c: dict(family_counts[c]) for c in cells},
        "spread": stats,
        "within_cell_rmse_A": within,
        "between_cell_rmse_A": between,
        "n_profiles_feasible_on_all_eval_twins": n_all_feasible,
        "voltage_headroom": voltage_headroom,
        "criteria": criteria,
        "i_max_a": float(i_max_a),
        "operating_box": {
            "soc_start": opt.session.soc_start,
            "t0_c": opt.session.ambient_t0_c,
            "v_max": opt.session.v_max,
            "energy_fraction": opt.session.energy_fraction,
            "i_min_a": opt.space.i_min_a,
            "i_max_a": float(i_max_a),
            "n_evals_per_family": opt.n_evals_per_family,
        },
        "start_states": {c: states[c] for c in eval_cells if c in states},
        "checkpoints": {
            c: str((checkpoints or {}).get(c, paths.checkpoint(c if c in CELLS else "RW9")))
            for c in eval_cells
        },
        "provenance": provenance(
            "Final_Aging_based_setup_litmus",
            configs=["paths", "reward", "optimization", "degradation_fitted"],
            inputs=[gpbo_dir / c / "summary.json" for c in cells],
        ),
    }
    write_json(out / "summary.json", payload)
    write_json(out / "criteria.json", criteria)

    fam_rows = []
    for fid in FAMILY_ORDER:
        fam_rows.append([FAMILY_LABELS[fid]] + [str(family_counts[c][fid]) for c in cells])
    master_md = []
    for e in elite:
        p = e["params"]
        nm = e["native_replay"]
        master_md.append([
            e["source_cell"], str(e["rank"]), FAMILY_LABELS.get(e["family_id"], e["family_id"]),
            f"{nm['duration_min']:.6g}", f"{nm['q_total']:.8g}", f"{nm['reward']:.8g}",
            f"{nm['peak_v']:.8g}", f"{nm['i_mean']:.8g}", f"{nm['i_max']:.8g}",
            _param_short(p),
        ])
    cv_rows = []
    for name, s in stats.items():
        cv_rows.append([
            name, f"{s['mean']:.8g}", f"{s['std']:.8g}", f"{s['cv_pct']:.8g}",
            f"{s['min']:.8g}", f"{s['max']:.8g}",
        ])
    c1_rows = []
    for c in cells:
        d = criteria["C1"]["per_cell"][c]
        c1_rows.append([c, str(d["dominant_family"]), str(d["count"]), "PASS" if d["pass"] else "FAIL"])
    c3_rows = []
    for c in cells:
        d = criteria["C3"]["per_cell"][c]
        c3_rows.append([c, str(d["n_feasible_on_all_twins"]), str(d["of"]), "PASS" if d["pass"] else "FAIL"])

    report = f"""# Litmus analysis

GP-BO directory: `{gpbo_dir}`
i_max_a = {i_max_a}

## C1–C3 (pre-registered; not changed)

Verdict: **{criteria['verdict']}**

C1 (dominant family ≥ 4/6 per cell): **{'PASS' if criteria['C1']['pass'] else 'FAIL'}**

{_md_table(['Cell', 'Dominant family', 'Count in Top-6', 'C1'], c1_rows)}

C2 (CV of peak current across {len(elite)} elites < 15%): **{'PASS' if criteria['C2']['pass'] else 'FAIL'}**
CV = {criteria['C2']['cv_pct_peak_current']} %
mean = {criteria['C2']['mean']} A, std = {criteria['C2']['std']} A, min = {criteria['C2']['min']} A, max = {criteria['C2']['max']} A

C3 (≥ 4/6 elites feasible on all eval twins, per cell): **{'PASS' if criteria['C3']['pass'] else 'FAIL'}**

{_md_table(['Cell', 'n feasible on all twins', 'of', 'C3'], c3_rows)}

Profiles feasible on all eval twins (global): {n_all_feasible} / {len(elite)}

## Level 1 — charging family in each Top-6

{_md_table(['Family', *list(cells)], fam_rows)}

## Level 2 — elite profiles (native-twin replay)

{_md_table(
    ['Cell', 'Rank', 'Family', 't (min)', 'Q_loss', 'Reward', 'V_peak', 'I_mean (A)', 'I_max (A)', 'Parameters'],
    master_md,
)}

## Parameter / outcome spread

{_md_table(['Quantity', 'Mean', 'Std', 'CV %', 'Min', 'Max'], cv_rows)}

Mean pairwise RMSE within a cell's Top-6:

{_md_table(['Cell', 'Mean within-cell RMSE (A)'], [[c, f"{within[c]}"] for c in cells])}

Mean pairwise RMSE between different cells: {between} A.
"""
    (out / "REPORT.md").write_text(report, encoding="utf-8")

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = np.arange(len(list(cells)))
    bottom = np.zeros(len(list(cells)))
    palette = {"cccv": "#2563eb", "two_step": "#16a34a",
               "three_step": "#ea580c", "pulsed": "#7c3aed"}
    for fid in FAMILY_ORDER:
        h = np.array([family_counts[c][fid] for c in cells], dtype=float)
        ax.bar(x, h, bottom=bottom, color=palette[fid], label=FAMILY_LABELS[fid], width=0.62)
        bottom += h
    ax.set_xticks(x)
    ax.set_xticklabels(list(cells))
    ax.set_ylabel("Count in Top-6")
    ax.set_ylim(0, k)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    savefig(fig, fig_dir / "family_counts.png")

    if native:
        apply_paper_style()
        fig, ax = plt.subplots(figsize=(9.2, 5.2))
        for row in native:
            t_min_arr = np.asarray(row["t_s"]) / 60.0
            ax.plot(t_min_arr, row["i_a"], color=CELL_COLORS.get(row["source_cell"], "#111"),
                    lw=1.6, alpha=0.75)
        for cell, col in CELL_COLORS.items():
            if cell in cells:
                ax.plot([], [], color=col, lw=2.4, label=cell)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Charge current (A)")
        ax.legend(frameon=False, ncol=4, loc="upper right")
        fig.tight_layout()
        savefig(fig, fig_dir / "current_overlay.png")

        apply_paper_style()
        nsrc = len(list(cells))
        nrows = 2 if nsrc > 1 else 1
        ncols = 2 if nsrc > 2 else max(1, nsrc)
        fig, axes = plt.subplots(nrows, ncols, figsize=(11.0, 8.2), sharey=True)
        axes_list = np.atleast_1d(axes).ravel()
        by_cell: Dict[str, List[Dict[str, Any]]] = {c: [] for c in cells}
        for row in native:
            by_cell[row["source_cell"]].append(row)
        for ax, cell in zip(axes_list, cells):
            for row in by_cell[cell]:
                t_min_arr = np.asarray(row["t_s"]) / 60.0
                ax.plot(t_min_arr, row["i_a"], lw=1.7, alpha=0.85,
                        label=f"P{row['rank']} {FAMILY_LABELS.get(row['family_id'], row['family_id'])}")
            ax.set_title(cell)
            ax.set_xlabel("Time (min)")
            ax.legend(fontsize=9, frameon=False, loc="best")
        axes_list[0].set_ylabel("Charge current (A)")
        fig.tight_layout()
        savefig(fig, fig_dir / "current_grid.png")

    if dist.size:
        apply_paper_style()
        fig, ax = plt.subplots(figsize=(9.6, 8.4))
        im = ax.imshow(dist, cmap="magma_r", origin="upper")
        ax.set_xticks(range(len(ids)))
        ax.set_yticks(range(len(ids)))
        ax.set_xticklabels(ids, rotation=90, fontsize=8)
        ax.set_yticklabels(ids, fontsize=8)
        for kk in range(1, n_cells_src):
            ax.axhline(kk * k - 0.5, color="white", lw=1.2)
            ax.axvline(kk * k - 0.5, color="white", lw=1.2)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("RMSE of I(t) (A), time-normalized")
        fig.tight_layout()
        savefig(fig, fig_dir / "similarity_24x24.png")

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.2, 8.6))
    im = ax.imshow(feas_grid, cmap="Greens", origin="upper", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(eval_cells)))
    ax.set_xticklabels(list(eval_cells))
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels(ids, fontsize=8)
    ax.set_xlabel("Evaluated on twin")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Feasible (1 = yes)")
    fig.tight_layout()
    savefig(fig, fig_dir / "cross_feasible.png")

    print(f"Wrote {out}", flush=True)
    print(f"Family counts: {{ {', '.join(repr(c)+': '+repr(dict(family_counts[c])) for c in cells)} }}", flush=True)
    print(f"Criteria verdict: {criteria['verdict']}", flush=True)
    print(f"C2 I_max CV: {criteria['C2']['cv_pct_peak_current']} %", flush=True)
    return payload
