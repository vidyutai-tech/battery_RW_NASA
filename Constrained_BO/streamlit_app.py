#!/usr/bin/env python3
"""Streamlit UI: set energy/operational constraints, run GP-BO + random search, download artifacts.

Launch from repo root:
    streamlit run Constrained_BO/streamlit_app.py
"""

from __future__ import annotations

import json
import os
import sys
import zipfile
from io import BytesIO
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(_ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import streamlit as st

from Constrained_BO.bayesian_optimizer import DEFAULT_ACQ_FUNC
from Constrained_BO.config import (
    ALL_CELLS,
    DEFAULT_ENERGY_FRACTION,
    MAX_DURATION_MIN,
    SOC_TARGET,
    energy_fraction_for,
)
from Constrained_BO.objective import (
    energy_required_j,
    full_capacity_joules,
)
from Constrained_BO.optimize_api import (
    DEGRADATION_CHART_METRICS,
    METRIC_COLS,
    PAPER_ACQ,
    PAPER_ELITE_TOP_K,
    PAPER_N_CALLS,
    PAPER_N_INITIAL,
    PAPER_N_RANDOM,
    PRIMARY_CHART_METRICS,
    baseline_rows_to_dataframe,
    build_baseline_comparison_rows,
    build_cell,
    comparison_dataframe,
    dataframe_to_excel_bytes,
    generate_degradation_report_figures,
    plot_baseline_bar_png,
    plot_profiles_png,
    results_to_dataframe,
    run_optimization,
    save_run_artifacts,
    winner_counts,
    winner_summary,
)
from Constrained_BO.profiles import DEFAULT_FAMILIES


st.set_page_config(
    page_title="Constrained Charging BO",
    page_icon="🔋",
    layout="wide",
)

st.title("Constrained charging profile optimization")
st.caption(
    "Hybrid reward **R = w_soc·ΔSoC − w_qloss·Q_total − w_time·t^z** "
    "(not Q_loss alone). Under equal-energy, claim **reward / duration / vs CC** — "
    "energy and Q_total bars are expected to look similar."
)

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.header("Presets")
    paper_preset = st.checkbox(
        "Paper defaults (BO-favored, fair)",
        value=True,
        help=f"EI, n_calls={PAPER_N_CALLS}, n_random={PAPER_N_RANDOM}, "
             f"inject top-{PAPER_ELITE_TOP_K} random elites into GP-BO warm-start.",
    )

    st.header("Cell & constraints")
    nasa_cells = [c for c in ALL_CELLS if c != "LFP"]
    cell_id = st.selectbox("Cell", nasa_cells, index=0)

    constraint_mode = st.radio(
        "Constraint mode",
        ("Energy delivery", "SoC target"),
        help="Energy mode stops when ∫V·I dt reaches the required joules.",
    )
    soc_mode = constraint_mode == "SoC target"

    default_efrac = energy_fraction_for(cell_id)
    energy_fraction = st.slider(
        "Energy fraction of pack",
        min_value=0.10,
        max_value=0.80,
        value=float(default_efrac),
        step=0.05,
        disabled=soc_mode,
        help=f"Default for {cell_id}: {default_efrac:.0%} "
             f"(global default {DEFAULT_ENERGY_FRACTION:.0%})",
    )
    soc_target = st.slider(
        "SoC target",
        min_value=0.50,
        max_value=0.95,
        value=float(SOC_TARGET),
        step=0.05,
        disabled=not soc_mode,
    )
    max_duration_min = st.number_input(
        "Max duration (min)",
        min_value=10.0,
        max_value=300.0,
        value=float(MAX_DURATION_MIN),
        step=10.0,
    )

    st.header("Optimizer budgets")
    families = st.multiselect(
        "Profile families",
        options=list(DEFAULT_FAMILIES),
        default=list(DEFAULT_FAMILIES),
    )
    _n_calls_def = PAPER_N_CALLS if paper_preset else 40
    _n_init_def = PAPER_N_INITIAL if paper_preset else 10
    _n_rand_def = PAPER_N_RANDOM if paper_preset else 80
    _acq_opts = ["EI", "PI", "LCB"]
    _acq_def = PAPER_ACQ if paper_preset else DEFAULT_ACQ_FUNC

    n_calls = st.number_input("GP-BO evaluations / family", 10, 300, int(_n_calls_def), 5)
    n_initial = st.number_input("GP-BO warm-start points", 3, 50, int(_n_init_def), 1)
    acq_func = st.selectbox(
        "Acquisition",
        _acq_opts,
        index=_acq_opts.index(_acq_def) if _acq_def in _acq_opts else 0,
    )
    n_random = st.number_input("Random-search samples / family", 10, 400, int(_n_rand_def), 10)
    inject_elites = st.checkbox(
        "Inject random elites into GP-BO warm-start",
        value=bool(paper_preset),
        help="When both run: random search first, then GP-BO warm-started with top feasible points.",
    )
    seed = st.number_input("Seed", 0, 10_000, 42, 1)
    device = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0)

    st.header("Hybrid reward weights")
    w_soc = st.number_input("w_soc (ΔSoC)", 0.0, 10.0, 1.0, 0.1)
    w_qloss = st.number_input("w_qloss (Q_loss index)", 0.0, 10.0, 1.0, 0.1)
    w_time = st.number_input("w_time (t^z)", 0.0, 10.0, 0.1, 0.05)
    z = st.number_input("z (time / calendar exponent)", 0.1, 1.5, 0.55, 0.05)

    run_both = st.checkbox("Run both GP-BO and random search", value=True)
    methods_only = st.multiselect(
        "Methods (if not both)",
        ["gp_bo", "random_search"],
        default=["gp_bo", "random_search"],
        disabled=run_both,
    )

    run_clicked = st.button("Run optimization", type="primary", use_container_width=True)

# --------------------------------------------------------------------------- #
# Preview
# --------------------------------------------------------------------------- #

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("Cell", cell_id)
with col_b:
    st.metric("Mode", "energy" if not soc_mode else "soc")
with col_c:
    if soc_mode:
        st.metric("SoC target", f"{soc_target:.0%}")
    else:
        st.metric("Energy fraction", f"{energy_fraction:.0%}")

st.info(
    "Q_calendar / Q_cyclic / Q_total are a **Relative Capacity-Loss Index** "
    "(ranking signal), not calibrated Capacity Fade (%)."
)

# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #

if run_clicked:
    if not families:
        st.error("Select at least one profile family.")
        st.stop()

    methods = ["gp_bo", "random_search"] if run_both else list(methods_only)
    if not methods:
        st.error("Select at least one method.")
        st.stop()

    with st.spinner("Loading cell / BDT checkpoint…"):
        try:
            cell = build_cell(
                cell_id,
                energy_fraction=None if soc_mode else float(energy_fraction),
                soc_mode=soc_mode,
                soc_target=float(soc_target) if soc_mode else None,
                max_duration_min=float(max_duration_min),
            )
        except Exception as exc:
            st.exception(exc)
            st.stop()

    if cell.constraint_mode == "energy":
        e_full = full_capacity_joules(cell.q_rated_as, cell.v_nom)
        e_req = energy_required_j(cell.q_rated_as, cell.energy_fraction, cell.v_nom)
        st.write(
            f"**Energy target:** {cell.energy_fraction:.0%} of {e_full:.0f} J "
            f"→ **{e_req:.0f} J** · V_nom={cell.v_nom:.3f} V · "
            f"start SoC={cell.start_state.get('soc', 0):.0%}"
        )
    else:
        st.write(
            f"**SoC target:** {cell.soc_target:.0%} · "
            f"start SoC={cell.start_state.get('soc', 0):.0%} · "
            f"V_nom={cell.v_nom:.3f} V"
        )

    reward_kwargs = {
        "reward_mode": "hybrid_qloss",
        "w_soc": float(w_soc),
        "w_qloss": float(w_qloss),
        "w_time": float(w_time),
        "w_temperature": 1.0,
        "z": float(z),
    }

    payloads: dict = {}
    results_map: dict = {}
    simulator = None
    elite_histories = None

    ordered = list(methods)
    if inject_elites and "gp_bo" in ordered and "random_search" in ordered:
        ordered = ["random_search", "gp_bo"]

    progress = st.progress(0.0, text="Starting…")
    for i, method in enumerate(ordered):
        label = "GP-BO" if method == "gp_bo" else "Random search"
        progress.progress(i / max(len(ordered), 1), text=f"Running {label}…")
        try:
            kwargs = dict(
                method=method,
                families=families,
                device=device,
                seed=int(seed),
                reward_mode="hybrid_qloss",
                w_soc=float(w_soc),
                w_qloss=float(w_qloss),
                w_time=float(w_time),
                z=float(z),
                n_calls=int(n_calls),
                n_initial=int(n_initial),
                acq_func=acq_func,
                n_random=int(n_random),
                simulator=simulator,
            )
            if method == "gp_bo" and elite_histories is not None:
                kwargs["elite_histories"] = elite_histories
                kwargs["elite_top_k"] = PAPER_ELITE_TOP_K
            payload, family_results, simulator = run_optimization(cell, **kwargs)
        except Exception as exc:
            progress.empty()
            st.exception(exc)
            st.stop()
        payloads[method] = payload
        results_map[method] = family_results
        if method == "random_search" and inject_elites:
            elite_histories = {
                fid: (entry.get("history") or [])
                for fid, entry in family_results.items()
            }
    progress.progress(1.0, text="Building CC baselines & degradation report…")

    with st.spinner("CC baselines + degradation figures…"):
        baseline_rows = build_baseline_comparison_rows(
            cell,
            simulator,
            bo_payload=payloads.get("gp_bo"),
            random_payload=payloads.get("random_search"),
            reward_kwargs=reward_kwargs,
        )
        deg_cache = (
            _ROOT / "Constrained_BO" / "results" / "ui_runs" / cell.cell_id / "degradation_report"
        )
        bo_json_tmp = None
        if "gp_bo" in payloads:
            deg_cache.mkdir(parents=True, exist_ok=True)
            bo_json_tmp = deg_cache / "_gp_bo_for_figs.json"
            bo_json_tmp.write_text(json.dumps(payloads["gp_bo"], indent=2))
        deg_paths = generate_degradation_report_figures(
            deg_cache,
            results_path=bo_json_tmp,
            device=device,
        )

    progress.empty()

    st.session_state["ui_cell"] = cell
    st.session_state["ui_payloads"] = payloads
    st.session_state["ui_results"] = results_map
    st.session_state["ui_baseline_rows"] = baseline_rows
    st.session_state["ui_deg_paths"] = {k: str(v) for k, v in deg_paths.items()}
    st.session_state["ui_reward_kwargs"] = reward_kwargs
    st.session_state["ui_device"] = device

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #

payloads = st.session_state.get("ui_payloads")
results_map = st.session_state.get("ui_results")
cell = st.session_state.get("ui_cell")
baseline_rows = st.session_state.get("ui_baseline_rows") or []
deg_paths_raw = st.session_state.get("ui_deg_paths") or {}

if not payloads:
    st.markdown(
        "Configure constraints in the sidebar and click **Run optimization**. "
        "Typical wall time: several minutes depending on budgets and device."
    )
    with st.expander("Degradation model (Figs 1–2, no run required)", expanded=False):
        cache = _ROOT / "Constrained_BO" / "results" / "ui_runs" / "_static" / "degradation_report"
        if st.button("Generate calendar / cyclic figures"):
            with st.spinner("Generating…"):
                paths = generate_degradation_report_figures(cache)
                st.session_state["ui_static_deg"] = {k: str(v) for k, v in paths.items()}
        static = st.session_state.get("ui_static_deg") or {}
        repo_deg = _ROOT / "Constrained_BO" / "results" / "degradation_report"
        for key, title in (
            ("fig1_calendar_contour", "Figure 1 — Calendar contour"),
            ("fig2_cyclic_curves", "Figure 2 — Cyclic curves"),
        ):
            p = Path(static[key]) if static.get(key) else repo_deg / f"{key}.png"
            if p.is_file():
                st.subheader(title)
                st.image(str(p), use_container_width=True)
                st.download_button(
                    f"Download {key}.png",
                    data=p.read_bytes(),
                    file_name=f"{key}.png",
                    mime="image/png",
                    key=f"pre_{key}",
                )
else:
    bo_payload = payloads.get("gp_bo")
    rnd_payload = payloads.get("random_search")

    if bo_payload and rnd_payload:
        df = comparison_dataframe(bo_payload, rnd_payload)
    elif bo_payload:
        df = results_to_dataframe(bo_payload, method_label="GP-BO")
    else:
        df = results_to_dataframe(rnd_payload, method_label="Random search")

    st.subheader("Winner summary (feasible total reward)")
    wsum = winner_summary(df)
    counts = winner_counts(wsum)
    c1, c2, c3 = st.columns(3)
    c1.metric("GP-BO family wins", counts["GP-BO"])
    c2.metric("Random family wins", counts["Random search"])
    c3.metric("Ties", counts["Tie"])
    if not wsum.empty:
        st.dataframe(wsum, use_container_width=True, hide_index=True)

    display_cols = [c for c, _ in METRIC_COLS if c in df.columns]
    rename = {c: lab for c, lab in METRIC_COLS}
    st.subheader("Comparison table")
    st.dataframe(
        df[display_cols].rename(columns=rename),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Best profiles")
    png_bytes = {}
    plot_cols = st.columns(max(len(results_map), 1))
    for col, (method, family_results) in zip(plot_cols, results_map.items()):
        label = "GP-BO" if method == "gp_bo" else "Random search"
        title = f"{label}, hybrid_qloss"
        if cell.constraint_mode == "energy" and cell.energy_fraction is not None:
            title += f", energy ≥ {cell.energy_fraction:.0%} of pack"
        try:
            png = plot_profiles_png(family_results, cell, title_suffix=title)
            png_bytes[method] = png
            col.image(png, caption=label, use_container_width=True)
            col.download_button(
                f"Download {label} PNG",
                data=png,
                file_name=f"{cell.cell_id}_{method}_best_profiles.png",
                mime="image/png",
                key=f"dl_png_{method}",
            )
        except Exception as exc:
            col.warning(f"Could not plot {label}: {exc}")

    st.subheader("Metric comparison")
    st.caption(
        "Equal-energy runs deliver nearly the same joules by design — "
        "use **Total reward** and **Duration** to discriminate methods."
    )
    chart_tabs = st.tabs([lab for _, lab in PRIMARY_CHART_METRICS])
    for tab, (key, lab) in zip(chart_tabs, PRIMARY_CHART_METRICS):
        with tab:
            if key not in df.columns:
                st.write("Metric not available.")
                continue
            pivot = df.pivot_table(
                index="family_label",
                columns="method",
                values=key,
                aggfunc="first",
            )
            st.bar_chart(pivot, stack=False)
            if pivot.shape[1] == 2:
                cols = list(pivot.columns)
                delta = pivot[cols[1]] - pivot[cols[0]]
                delta_df = pivot.copy()
                delta_df[f"Δ ({cols[1]} − {cols[0]})"] = delta
                st.dataframe(delta_df, use_container_width=True)
            if key == "energy_delivered_j":
                st.caption("Nearly identical bars are expected under energy-stop constraints.")
            st.caption(lab)

    with st.expander("Degradation index (Q_calendar / Q_cyclic / Q_total)"):
        st.caption(
            "Relative Capacity-Loss Index — secondary ranking signal, not % fade."
        )
        deg_tabs = st.tabs([lab for _, lab in DEGRADATION_CHART_METRICS])
        for tab, (key, lab) in zip(deg_tabs, DEGRADATION_CHART_METRICS):
            with tab:
                if key not in df.columns:
                    continue
                pivot = df.pivot_table(
                    index="family_label",
                    columns="method",
                    values=key,
                    aggfunc="first",
                )
                st.bar_chart(pivot, stack=False)
                st.caption(lab)

    st.subheader("CC baselines vs optimized (hima-style)")
    if baseline_rows:
        bdf = baseline_rows_to_dataframe(baseline_rows)
        st.dataframe(bdf, use_container_width=True, hide_index=True)
        baseline_pngs = {}
        bcols = st.columns(3)
        for col, (key, ylabel, title, fname) in zip(
            bcols,
            (
                ("total_reward", "Total reward", "Reward comparison", "reward_comparison.png"),
                ("duration_min", "Duration (min)", "Time comparison", "time_comparison.png"),
                ("peak_temperature", "Peak T (°C)", "Temperature comparison", "temperature_comparison.png"),
            ),
        ):
            try:
                png = plot_baseline_bar_png(
                    baseline_rows, value_key=key, ylabel=ylabel, title=title,
                )
                baseline_pngs[fname] = png
                col.image(png, caption=title, use_container_width=True)
                col.download_button(
                    f"Download {fname}",
                    data=png,
                    file_name=f"{cell.cell_id}_{fname}",
                    mime="image/png",
                    key=f"dl_base_{fname}",
                )
            except Exception as exc:
                col.warning(str(exc))
        st.session_state["ui_baseline_pngs"] = baseline_pngs
    else:
        st.write("No baseline rows (run optimization to populate).")

    st.subheader("Degradation report")
    deg_paths = {k: Path(v) for k, v in deg_paths_raw.items()}
    repo_deg = _ROOT / "Constrained_BO" / "results" / "degradation_report"
    fig_specs = [
        ("fig1_calendar_contour", "Figure 1 — Calendar degradation contour"),
        ("fig2_cyclic_curves", "Figure 2 — Cyclic degradation curves"),
        ("fig3_cumulative_degradation", "Figure 3 — Cumulative degradation (best profiles)"),
        ("fig4_equal_energy_table", "Figure 4 — Equal-energy comparison table"),
    ]
    for key, title in fig_specs:
        p = deg_paths.get(key)
        if p is None or not Path(p).is_file():
            alt = repo_deg / f"{key}.png"
            p = alt if alt.is_file() else None
        if p is None:
            st.write(f"*{title}: not available yet.*")
            continue
        st.markdown(f"**{title}**")
        st.image(str(p), use_container_width=True)
        st.download_button(
            f"Download {Path(p).name}",
            data=Path(p).read_bytes(),
            file_name=Path(p).name,
            mime="image/png",
            key=f"dl_deg_{key}",
        )
    csv4 = deg_paths.get("fig4_equal_energy_table_csv")
    if csv4 and Path(csv4).is_file():
        st.dataframe(pd.read_csv(csv4), use_container_width=True, hide_index=True)

    st.subheader("Downloads")
    excel_bytes = dataframe_to_excel_bytes(df)
    st.download_button(
        "Download comparison Excel (.xlsx)",
        data=excel_bytes,
        file_name=f"{cell.cell_id}_bo_vs_random_comparison.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.download_button(
        "Download comparison CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"{cell.cell_id}_bo_vs_random_comparison.csv",
        mime="text/csv",
    )
    if baseline_rows:
        st.download_button(
            "Download baseline comparison CSV",
            data=baseline_rows_to_dataframe(baseline_rows).to_csv(index=False).encode("utf-8"),
            file_name=f"{cell.cell_id}_baseline_comparison.csv",
            mime="text/csv",
            key="dl_base_csv",
        )

    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{cell.cell_id}_comparison.xlsx", excel_bytes)
        zf.writestr(f"{cell.cell_id}_comparison.csv", df.to_csv(index=False))
        for method, png in png_bytes.items():
            zf.writestr(f"{cell.cell_id}_{method}_best_profiles.png", png)
        for method, payload in payloads.items():
            zf.writestr(
                f"{cell.cell_id}_{method}_results.json",
                json.dumps(payload, indent=2),
            )
        for fname, png in (st.session_state.get("ui_baseline_pngs") or {}).items():
            zf.writestr(f"{cell.cell_id}_{fname}", png)
        for key, p in deg_paths.items():
            if Path(p).is_file() and Path(p).suffix == ".png":
                zf.writestr(f"degradation_report/{Path(p).name}", Path(p).read_bytes())
    zip_buf.seek(0)
    st.download_button(
        "Download all artifacts (ZIP)",
        data=zip_buf.getvalue(),
        file_name=f"{cell.cell_id}_optimization_artifacts.zip",
        mime="application/zip",
    )

    with st.expander("Save to disk under Constrained_BO/results/ui_runs"):
        if st.button("Write artifacts to results/ui_runs"):
            out = _ROOT / "Constrained_BO" / "results" / "ui_runs" / cell.cell_id
            if bo_payload is None or rnd_payload is None:
                out.mkdir(parents=True, exist_ok=True)
                for method, payload in payloads.items():
                    (out / f"{method}_results.json").write_text(
                        json.dumps(payload, indent=2)
                    )
                for method, png in png_bytes.items():
                    (out / f"{method}_best_profiles.png").write_bytes(png)
                (out / "comparison.xlsx").write_bytes(excel_bytes)
                st.success(f"Wrote {out}")
            else:
                paths = save_run_artifacts(
                    out,
                    bo_payload=bo_payload,
                    random_payload=rnd_payload,
                    bo_results=results_map["gp_bo"],
                    random_results=results_map["random_search"],
                    cell=cell,
                    comparison_df=df,
                    baseline_rows=baseline_rows,
                    device=st.session_state.get("ui_device", "auto"),
                )
                st.success(
                    "Wrote:\n" + "\n".join(f"- `{p}`" for p in paths.values())
                )
