"""
Output layout for the charging-profile pipeline.

Three top-level folders under ``outputs/charging_opt/``:

    models/     — .npz, .pt references, per-stage ``registry.json`` with metrics
    plots/      — PNG figures and tabular exports (CSV/JSON) per stage
    registry/   — master ``charging_opt_registry.json`` + ``artifacts_manifest.json``

Stage subfolders (patent-aligned):

    stage1_state_estimation   — OCV–SoC, capacity fade, BDT drift margins
    stage2_reward_diagnostic  — CC sweep tables (objective sanity checks)
    stage3_optimization     — Bayesian optimization results
"""

from __future__ import annotations

from pathlib import Path

# Root
CHARGING_OPT = "outputs/charging_opt"
MODELS = f"{CHARGING_OPT}/models"
PLOTS = f"{CHARGING_OPT}/plots"
REGISTRY = f"{CHARGING_OPT}/registry"
SNAPSHOTS = f"{CHARGING_OPT}/snapshots"

# Stage 1 — state estimation (SoC curves + drift)
STAGE1 = "stage1_state_estimation"
STAGE1_MODELS = f"{MODELS}/{STAGE1}"
STAGE1_PLOTS = f"{PLOTS}/{STAGE1}"
STAGE1_DRIFT = "stage1_drift"
STAGE1_DRIFT_MODELS = f"{MODELS}/{STAGE1_DRIFT}"
STAGE1_DRIFT_PLOTS = f"{PLOTS}/{STAGE1_DRIFT}"

# Stage 2 — reward / objective diagnostics
STAGE2 = "stage2_reward_diagnostic"
STAGE2_MODELS = f"{MODELS}/{STAGE2}"
STAGE2_PLOTS = f"{PLOTS}/{STAGE2}"

# Stage 3 — profile optimization
STAGE3 = "stage3_optimization"
STAGE3_MODELS = f"{MODELS}/{STAGE3}"
STAGE3_PLOTS = f"{PLOTS}/{STAGE3}"


def charging_opt_root(root: Path | None = None) -> Path:
    if root is None:
        root = Path(__file__).resolve().parents[1]
    return root / CHARGING_OPT


def ensure_layout(root: Path | None = None) -> dict[str, Path]:
    """Create all stage directories; return resolved path map."""
    base = charging_opt_root(root)
    dirs = {
        "models": base / "models",
        "plots": base / "plots",
        "registry": base / "registry",
        "snapshots": base / "snapshots",
        STAGE1 + "_models": base / "models" / STAGE1,
        STAGE1 + "_plots": base / "plots" / STAGE1,
        STAGE1_DRIFT + "_models": base / "models" / STAGE1_DRIFT,
        STAGE1_DRIFT + "_plots": base / "plots" / STAGE1_DRIFT,
        STAGE2 + "_models": base / "models" / STAGE2,
        STAGE2 + "_plots": base / "plots" / STAGE2,
        STAGE3 + "_models": base / "models" / STAGE3,
        STAGE3 + "_plots": base / "plots" / STAGE3,
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def rel(path: str) -> str:
    """Normalize to forward-slash relative path from repo root."""
    return path.replace("\\", "/")
