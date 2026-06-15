"""Helpers for writing artifacts when shared output files are not owned by the current user."""

from __future__ import annotations

import os
from pathlib import Path


def current_user() -> str:
    return os.getenv("USER") or os.getenv("LOGNAME") or "user"


def dir_is_writable(path: Path) -> bool:
    """Return True if the directory exists (or can be created) and accepts new files."""
    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f".write_probe_{current_user()}"
        probe.touch()
        probe.unlink()
        return True
    except OSError:
        return False


def user_stage3_root(repo_root: Path, user: str | None = None) -> Path:
    """Dedicated stage-3 output tree owned by the current user."""
    user = user or current_user()
    return repo_root / "outputs" / "charging_opt_user" / user / "stage3_optimization"


def fresh_user_stage3_base(repo_root: Path, user: str | None = None) -> Path:
    """New timestamped stage-3 directory when the default tree is not writable."""
    from datetime import datetime

    user = user or current_user()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return repo_root / "outputs" / "charging_opt_user" / user / f"stage3_{stamp}"


def user_fallback_dir(
    original_dir: Path,
    user: str | None = None,
    *,
    repo_root: Path | None = None,
) -> Path:
    """
    Writable directory when *original_dir* (or its parent) is root-owned.

    Tries ``<parent>/<name>_<user>`` when the parent is writable; otherwise a
    fresh timestamped tree under ``outputs/charging_opt_user/<user>/``.
    """
    user = user or current_user()
    original_dir = Path(original_dir)
    parent = original_dir.parent

    if dir_is_writable(parent):
        alt = parent / f"{original_dir.name}_{user}"
        alt.mkdir(parents=True, exist_ok=True)
        return alt

    if repo_root is not None:
        base = fresh_user_stage3_base(repo_root, user)
        alt = base / original_dir.name
        alt.mkdir(parents=True, exist_ok=True)
        return alt

    alt = Path("/tmp") / f"charging_opt_{user}" / original_dir.name
    alt.mkdir(parents=True, exist_ok=True)
    return alt


def resolve_writable_path(
    path: Path,
    *,
    suffix_user: bool = True,
    repo_root: Path | None = None,
) -> Path:
    """
    Return a path the current process can write.

    If the target directory is not writable (e.g. owned by root), redirect to
    a user-owned sibling directory (``models`` → ``models_<user>``).

    If only the target *file* is not writable but its directory is, append
    ``_<user>`` before the suffix.
    """
    path = Path(path)
    user = current_user()
    path.parent.mkdir(parents=True, exist_ok=True)

    if not dir_is_writable(path.parent):
        alt_dir = user_fallback_dir(path.parent, user, repo_root=repo_root)
        alt = alt_dir / path.name
        print(
            f"NOTE: {path.parent} is not writable (owned by another user).\n"
            f"      Writing to {alt_dir}/ instead.\n"
            f"      Or run: sudo bash scripts/fix_output_permissions.sh {user}"
        )
        return alt

    if path.exists() and not os.access(path, os.W_OK):
        if suffix_user:
            alt = path.with_name(f"{path.stem}_{user}{path.suffix}")
        else:
            alt = path.with_name(f"{path.stem}.new{path.suffix}")
        print(
            f"NOTE: cannot overwrite {path} (owned by another user).\n"
            f"      Writing to {alt} instead."
        )
        return alt

    return path


def resolve_stage3_family_dirs(
    repo_root: Path,
    *,
    out_dir: Path | None = None,
) -> tuple[Path, Path]:
    """
    Resolve (models_dir, plots_dir) for multi-family optimization.

    *out_dir* — optional base; creates ``models/`` and ``plots/profile_families/``.
    Falls back to a writable sibling tree when the requested directory is root-owned.
    """
    repo_root = Path(repo_root)
    if out_dir is not None:
        base = Path(out_dir)
        models = base / "models"
        plots = base / "plots" / "profile_families"
        if not dir_is_writable(models):
            alt_base = fresh_user_stage3_base(repo_root)
            print(
                f"NOTE: {base} is not writable — using fresh run directory:\n"
                f"  {alt_base}"
            )
            base = alt_base
            models = base / "models"
            plots = base / "plots" / "profile_families"
        models.mkdir(parents=True, exist_ok=True)
        plots.mkdir(parents=True, exist_ok=True)
        return models, plots

    from charging_opt import paths as P

    models = (repo_root / P.STAGE3_MODELS).resolve()
    plots = (repo_root / P.STAGE3_PLOTS / "profile_families").resolve()
    models.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    probe = models / "family_optimization_results.json"
    if (probe.exists() and not os.access(probe, os.W_OK)) or not dir_is_writable(models):
        base = user_stage3_root(repo_root)
        models = base / "models"
        plots = base / "plots" / "profile_families"
        models.mkdir(parents=True, exist_ok=True)
        plots.mkdir(parents=True, exist_ok=True)
        print(
            f"NOTE: switching to user-owned output directory:\n"
            f"  models -> {models}\n"
            f"  plots  -> {plots}"
        )
    return models, plots


def resolve_stage3_pareto_dirs(
    repo_root: Path,
    *,
    out_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Return (models_dir, pareto_plots_dir) for Priority-3 artifacts."""
    if out_dir is not None:
        models, _ = resolve_stage3_family_dirs(repo_root, out_dir=out_dir)
        pareto_plots = models.parent / "plots" / "pareto"
        pareto_plots.mkdir(parents=True, exist_ok=True)
        return models, pareto_plots

    models, _ = resolve_stage3_family_dirs(repo_root)
    pareto_plots = models.parent / "plots" / "pareto"
    pareto_plots.mkdir(parents=True, exist_ok=True)
    return models, pareto_plots


def resolve_visualization_dir(
    repo_root: Path,
    out_dir: Path | None = None,
) -> Path:
    """
    Writable directory for publication figures (``outputs/visualization``).

    Falls back to ``outputs/charging_opt_user/<USER>/visualization`` when the
    shared tree is root-owned or existing PNGs cannot be overwritten.
    """
    import os

    repo_root = Path(repo_root)
    requested = Path(out_dir) if out_dir is not None else repo_root / "outputs" / "visualization"
    requested.mkdir(parents=True, exist_ok=True)

    user = current_user()
    fallback = repo_root / "outputs" / "charging_opt_user" / user / "visualization"

    def _redirect(reason: str) -> Path:
        fallback.mkdir(parents=True, exist_ok=True)
        print(
            f"NOTE: {reason}\n"
            f"      Writing figures to {fallback}/ instead.\n"
            f"      Or run: sudo bash scripts/fix_output_permissions.sh {user}"
        )
        return fallback

    if not dir_is_writable(requested):
        return _redirect(f"{requested} is not writable")

    for png in requested.glob("fig*.png"):
        if not os.access(png, os.W_OK):
            return _redirect(f"cannot overwrite {png} (owned by another user)")

    return requested
