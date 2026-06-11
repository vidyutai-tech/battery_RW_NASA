"""
Canonical artifact paths for the charging-profile pipeline.

Layout::

    outputs/charging_opt/
        models/
            stage1_state_estimation/   ocv_soc_curve.npz, capacity_fade.npz, registry.json
            stage1_drift/              conformal_margins.npz, registry.json  (drift margins)
            stage3_optimization/       optimization_result.json, best_session.json, registry.json
        plots/
            stage1_state_estimation/   ocv_soc_curve.png, capacity_fade.png
            stage1_drift/              drift_vs_horizon.png, per_action_rmse.png
            stage2_reward_diagnostic/  cc_sweep.json, cc_sweep_*.png
            stage3_optimization/       best_profile.png, bo_convergence.png
        registry/
            charging_opt_registry.json
            artifacts_manifest.json
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from charging_opt import paths as P

# Paths every downstream script expects (relative to repo root).
CANONICAL = {
    "bdt_source": "outputs/twin_source/20260610_111409/twin_source_RW9.pt",
    "bdt_finetune_rw10_20": "outputs/finetune_two_stage_RW10/registry/finetune_RW10_frac0.20.pt",
    "bdt_finetune_rw10_40": "outputs/finetune_two_stage_RW10/registry/finetune_RW10_frac0.40.pt",
    "bdt_finetune_rw10_60": "outputs/finetune_two_stage_RW10/registry/finetune_RW10_frac0.60.pt",
    # Stage 1 — SoC / capacity
    "ocv_curve": f"{P.STAGE1_MODELS}/ocv_soc_curve.npz",
    "capacity_fade": f"{P.STAGE1_MODELS}/capacity_fade.npz",
    "stage1_registry": f"{P.STAGE1_MODELS}/registry.json",
    # Stage 1 — drift (stored under models/stage1_drift for binary artifacts)
    "conformal_margins": f"{P.STAGE1_DRIFT_MODELS}/conformal_margins.npz",
    "drift_registry": f"{P.STAGE1_DRIFT_MODELS}/registry.json",
    # Master registry
    "master_registry": f"{P.REGISTRY}/charging_opt_registry.json",
}

OPTIONAL = {
    "lifetime_bo_result": f"{P.STAGE3_MODELS}/optimization_result.json",
    "stage3_registry": f"{P.STAGE3_MODELS}/registry.json",
    "plot_ocv_soc": f"{P.STAGE1_PLOTS}/ocv_soc_curve.png",
    "plot_capacity_fade": f"{P.STAGE1_PLOTS}/capacity_fade.png",
    "plot_drift": f"{P.STAGE1_DRIFT_PLOTS}/drift_vs_horizon.png",
    "plot_per_action": f"{P.STAGE1_DRIFT_PLOTS}/per_action_rmse.png",
    "cc_sweep_json": f"{P.STAGE2_PLOTS}/cc_sweep.json",
    "plot_cc_sweep": f"{P.STAGE2_PLOTS}/cc_sweep_sei_and_time.png",
    "plot_cc_tradeoff": f"{P.STAGE2_PLOTS}/cc_sweep_tradeoff.png",
    "plot_best_profile": f"{P.STAGE3_PLOTS}/best_profile.png",
    "plot_bo_convergence": f"{P.STAGE3_PLOTS}/bo_convergence.png",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve(rel_path: str, root: Optional[Path] = None) -> Path:
    return (root or _repo_root()) / rel_path


def resolve_bdt_ckpt(
    ckpt: str | Path | None = None,
    *,
    root: Optional[Path] = None,
) -> Path:
    """Return an existing BDT checkpoint; fall back to newest ``twin_source_RW9.pt``."""
    root = root or _repo_root()
    candidates: list[Path] = []
    if ckpt is not None:
        path = Path(ckpt)
        candidates.append(path if path.is_absolute() else root / path)
    candidates.append(_resolve(CANONICAL["bdt_source"], root))

    for path in candidates:
        if path.is_file():
            return path

    twin_root = root / "outputs" / "twin_source"
    found = sorted(twin_root.glob("*/twin_source_RW9.pt")) if twin_root.is_dir() else []
    if found:
        return found[-1]

    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"No BDT checkpoint found. Tried: {tried}. "
        f"Train with scripts/train_twin.py or pass --bdt_ckpt."
    )


def _file_info(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    data = path.read_bytes()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(data).hexdigest(),
        "modified_utc": datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
    }


def write_stage_registry(
    stage_models_dir: str,
    metrics: Dict[str, Any],
    *,
    root: Optional[Path] = None,
) -> Path:
    """Write ``registry.json`` inside a stage ``models/`` folder."""
    root = root or _repo_root()
    P.ensure_layout(root)
    out = root / stage_models_dir / "registry.json"
    payload = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage_dir": stage_models_dir,
        "metrics": metrics,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=float)
        f.write("\n")
    return out


def update_master_registry(*, root: Optional[Path] = None) -> Path:
    """Merge per-stage registries into ``registry/charging_opt_registry.json``."""
    root = root or _repo_root()
    P.ensure_layout(root)
    master: Dict[str, Any] = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "layout": {
            "models": P.MODELS,
            "plots": P.PLOTS,
            "registry": P.REGISTRY,
        },
        "stages": {},
        "external_models": {
            "bdt_source": CANONICAL["bdt_source"],
            "bdt_finetune_rw10_20": CANONICAL["bdt_finetune_rw10_20"],
            "bdt_finetune_rw10_40": CANONICAL["bdt_finetune_rw10_40"],
            "bdt_finetune_rw10_60": CANONICAL["bdt_finetune_rw10_60"],
        },
        "artifacts": {k: CANONICAL[k] for k in CANONICAL if not k.startswith("bdt_")},
    }
    stage_registries = [
        ("stage1_state_estimation", CANONICAL["stage1_registry"]),
        ("stage1_drift", CANONICAL["drift_registry"]),
        ("stage3_optimization", OPTIONAL["stage3_registry"]),
    ]
    for name, rel_path in stage_registries:
        p = _resolve(rel_path, root)
        if p.is_file():
            master["stages"][name] = json.loads(p.read_text(encoding="utf-8"))
        else:
            master["stages"][name] = {"exists": False, "path": rel_path}

    out = _resolve(CANONICAL["master_registry"], root)
    with out.open("w", encoding="utf-8") as f:
        json.dump(master, f, indent=2, default=float)
        f.write("\n")
    return out


def verify_artifacts(
    root: Optional[Path] = None,
    *,
    include_optional: bool = True,
) -> Tuple[bool, Dict[str, Any]]:
    root = root or _repo_root()
    P.ensure_layout(root)
    manifest: Dict[str, Any] = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(root),
        "layout": {"models": P.MODELS, "plots": P.PLOTS, "registry": P.REGISTRY},
        "required": {},
        "optional": {},
    }
    all_ok = True
    for name, rel in CANONICAL.items():
        if name.startswith("bdt_") or name == "master_registry":
            continue
        info = _file_info(_resolve(rel, root))
        manifest["required"][name] = info
        if not info["exists"]:
            all_ok = False
    if include_optional:
        for name, rel in OPTIONAL.items():
            manifest["optional"][name] = _file_info(_resolve(rel, root))
    manifest["all_required_present"] = all_ok
    return all_ok, manifest


def write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def snapshot_artifacts(
    dest_dir: Optional[Path] = None,
    root: Optional[Path] = None,
) -> Path:
    root = root or _repo_root()
    if dest_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        dest_dir = root / P.SNAPSHOTS / f"snapshot_{stamp}"
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    ok, manifest = verify_artifacts(root, include_optional=True)
    if not ok:
        missing = [k for k, v in manifest["required"].items() if not v["exists"]]
        raise FileNotFoundError(
            f"Cannot snapshot: missing required artifacts: {missing}"
        )

    copied: List[str] = []
    to_copy = list(CANONICAL.items()) + list(OPTIONAL.items())
    seen = set()
    for name, rel in to_copy:
        if rel in seen:
            continue
        seen.add(rel)
        src = _resolve(rel, root)
        if not src.is_file():
            continue
        rel_to_opt = src.relative_to(root)
        dst = dest_dir / rel_to_opt
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(str(rel_to_opt))

    manifest["snapshot_dir"] = str(dest_dir)
    manifest["copied_files"] = copied
    write_manifest(dest_dir / "manifest.json", manifest)

    pointer = root / P.CHARGING_OPT / "LATEST_SNAPSHOT.txt"
    pointer.write_text(str(dest_dir) + "\n", encoding="utf-8")
    return dest_dir
