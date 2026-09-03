"""Deterministic monotonicity tests for the hybrid degradation model.

Eq. (5) calendar and Eq. (8) / Table 7 cyclic:

- Calendar loss increases with SoC, temperature, and time.
- Cyclic loss increases with Ah throughput and with C-rate (2C–10C).
"""

from __future__ import annotations

import numpy as np

from Constrained_BO.hybrid_degradation import (
    CalendarDegradation,
    CyclicDegradation,
    HybridDegradationParameters,
)


def _calendar() -> CalendarDegradation:
    return CalendarDegradation(HybridDegradationParameters())


def _cyclic() -> CyclicDegradation:
    return CyclicDegradation(HybridDegradationParameters())


# --------------------------------------------------------------------------- #
# Calendar Eq. (5):
#   Q = A_cal*exp(B_cal*SoC)*exp(-(Ea_cal+C_cal*SoC)/(R*T))*t^z_cal
# --------------------------------------------------------------------------- #

def test_calendar_increases_with_soc():
    cal = _calendar()
    socs = [0.1, 0.3, 0.5, 0.7, 0.9]
    losses = [cal.q_loss(soc, temperature_c=25.0, t_hours=720.0) for soc in socs]
    for a, b in zip(losses, losses[1:]):
        assert b > a, f"calendar loss must strictly increase with SoC: {losses}"


def test_calendar_increases_with_temperature():
    cal = _calendar()
    temps_c = [0.0, 15.0, 25.0, 35.0, 45.0]
    losses = [cal.q_loss(soc=0.5, temperature_c=t, t_hours=720.0) for t in temps_c]
    for a, b in zip(losses, losses[1:]):
        assert b > a, f"calendar loss must strictly increase with temperature: {losses}"


def test_calendar_increases_with_time():
    cal = _calendar()
    hours = [1.0, 24.0, 168.0, 720.0, 2160.0]
    losses = [cal.q_loss(soc=0.5, temperature_c=25.0, t_hours=h) for h in hours]
    for a, b in zip(losses, losses[1:]):
        assert b > a, f"calendar loss must strictly increase with time: {losses}"


def test_calendar_incremental_is_nonnegative_and_consistent():
    """Cumulative differencing (used per-step in compute_step_degradation) must
    reproduce q_loss(t2) - q_loss(t1) and never go negative."""
    cal = _calendar()
    q1 = cal.q_loss(soc=0.5, temperature_c=25.0, t_hours=100.0)
    q2 = cal.q_loss(soc=0.5, temperature_c=25.0, t_hours=200.0)
    inc = cal.incremental(soc=0.5, temperature_c=25.0, t_hours_prev=100.0, t_hours_next=200.0)
    assert inc >= 0.0
    assert np.isclose(inc, q2 - q1)


# --------------------------------------------------------------------------- #
# Cyclic Eq. (8) / Table 7: Q = B(I)*exp(-Ea(I)/(R*T))*Ah^z(I)
# --------------------------------------------------------------------------- #

def test_cyclic_increases_with_ah_throughput():
    cyc = _cyclic()
    ah_values = [0.5, 1.0, 2.0, 5.0, 10.0]
    for c_rate in (0.5, 2.0, 6.0, 10.0):
        losses = [cyc.q_loss(ah, temperature_c=25.0, c_rate=c_rate) for ah in ah_values]
        for a, b in zip(losses, losses[1:]):
            assert b > a, (
                f"cyclic loss must strictly increase with Ah at C={c_rate}: {losses}"
            )


def test_cyclic_increases_with_crate_in_monotonic_region():
    """Table 7's published (B, Ea) coefficients are monotonic in C-rate from 2C to
    10C (verified numerically); this is the physically-expected "higher C-rate is
    worse" regime and is the one exercised here."""
    cyc = _cyclic()
    c_rates = [2.0, 4.0, 6.0, 8.0, 10.0]
    losses = [cyc.q_loss(ah_throughput=1.0, temperature_c=25.0, c_rate=c) for c in c_rates]
    for a, b in zip(losses, losses[1:]):
        assert b > a, f"cyclic loss must strictly increase with C-rate (2C-10C): {losses}"


def test_cyclic_high_crate_exceeds_low_crate_overall():
    """End-to-end sanity check across the full Table 7 grid: the highest tabulated
    C-rate (10C) must still degrade more than the lowest (C/2), even though the
    model is not monotonic in between (see next test)."""
    cyc = _cyclic()
    q_low = cyc.q_loss(ah_throughput=1.0, temperature_c=25.0, c_rate=0.5)
    q_high = cyc.q_loss(ah_throughput=1.0, temperature_c=25.0, c_rate=10.0)
    assert q_high > q_low


def test_cyclic_table7_known_nonmonotonic_dip_between_half_c_and_2c():
    """Documents a real, literature-sourced characteristic of Table 7: the fitted
    (B, Ea) pair at C/2 predicts *more* loss than at 2C (at matched Ah and T),
    before the trend reverses and increases monotonically from 2C to 10C. This is
    intentional fidelity to the published coefficients, not a code defect — see the
    NOTE above TABLE7_B/TABLE7_EA in hybrid_degradation.py. If this test starts
    failing after a coefficient change, update the docstring/README accordingly.
    """
    cyc = _cyclic()
    q_half_c = cyc.q_loss(ah_throughput=1.0, temperature_c=25.0, c_rate=0.5)
    q_2c = cyc.q_loss(ah_throughput=1.0, temperature_c=25.0, c_rate=2.0)
    assert q_half_c > q_2c


def test_cyclic_incremental_is_nonnegative_and_consistent():
    cyc = _cyclic()
    q1 = cyc.q_loss(ah_throughput=1.0, temperature_c=25.0, c_rate=2.0)
    q2 = cyc.q_loss(ah_throughput=3.0, temperature_c=25.0, c_rate=2.0)
    inc = cyc.incremental(ah_prev=1.0, ah_next=3.0, temperature_c=25.0, c_rate=2.0)
    assert inc >= 0.0
    assert np.isclose(inc, q2 - q1)


# --------------------------------------------------------------------------- #
# Session-level throughput metrics (EFC, mean/max C-rate, mean SoC)
# --------------------------------------------------------------------------- #

def _synthetic_session(current_a: float, n_seconds: int, q_rated_as: float = 7560.0):
    i = np.full(n_seconds, current_a, dtype=np.float64)
    soc0 = 0.2
    soc = np.clip(soc0 + np.cumsum(-i) / q_rated_as, 0.0, 1.0)
    return {
        "current_a": i,
        "temperature_c": np.full(n_seconds, 25.0, dtype=np.float64),
        "soc": soc,
        "q_rated_as": q_rated_as,
        "initial_state": {"soc": soc0},
    }


def test_efc_scales_linearly_with_ah_throughput():
    from Constrained_BO.hybrid_degradation import session_throughput_metrics

    q_rated_as = 7560.0
    session_1c_1h = _synthetic_session(-2.1, 3600, q_rated_as=q_rated_as)  # ~1C for 1h
    session_1c_2h = _synthetic_session(-2.1, 7200, q_rated_as=q_rated_as)

    m1 = session_throughput_metrics(session_1c_1h)
    m2 = session_throughput_metrics(session_1c_2h)

    assert m2["ah_throughput"] > m1["ah_throughput"]
    assert m2["efc"] > m1["efc"]
    # EFC convention: full charge+discharge = 2x rated Ah of throughput.
    q_rated_ah = q_rated_as / 3600.0
    assert np.isclose(m1["efc"], m1["ah_throughput"] / (2.0 * q_rated_ah))


def test_max_c_rate_geq_mean_c_rate():
    from Constrained_BO.hybrid_degradation import session_throughput_metrics

    i = np.concatenate([np.full(1800, -1.0), np.full(1800, -4.0)])
    session = {
        "current_a": i,
        "temperature_c": np.full(i.size, 25.0),
        "soc": np.clip(0.2 + np.cumsum(-i) / 7560.0, 0.0, 1.0),
        "q_rated_as": 7560.0,
        "initial_state": {"soc": 0.2},
    }
    m = session_throughput_metrics(session)
    assert m["max_c_rate"] >= m["nominal_c_rate"]


def test_session_step_ah_and_closed_form_totals_agree_on_pulsed():
    """Charge-only Ah and headline qloss_* must match session closed form."""
    from Constrained_BO.hybrid_degradation import (
        compute_session_degradation,
        compute_step_degradation,
    )

    parts = []
    for _ in range(5):
        parts.append(np.full(60, -2.0))
        parts.append(np.zeros(60))
    i = np.concatenate(parts)
    n = i.size
    q = 7560.0
    session = {
        "current_a": i,
        "temperature_c": np.full(n, 30.0),
        "soc": np.clip(0.2 + np.cumsum(-i) / q, 0.0, 1.0),
        "q_rated_as": q,
        "initial_state": {"soc": 0.2},
    }
    sess = compute_session_degradation(session)
    step = compute_step_degradation(session)
    assert np.isclose(sess["ah_throughput"], step["ah_throughput"])
    assert np.isclose(sess["qloss_calendar"], step["qloss_calendar"])
    assert np.isclose(sess["qloss_cyclic"], step["qloss_cyclic"])
    assert np.isclose(sess["qloss_total"], step["qloss_total"])
    # Constant T/SoC → step calendar sum matches closed form; cyclic uses
    # charge-only Ah so step cyclic sum matches session for constant C-rate.
    assert np.isclose(step["qloss_calendar_step_sum"], step["qloss_calendar"], rtol=1e-5)
    assert np.isclose(step["qloss_cyclic_step_sum"], step["qloss_cyclic"], rtol=1e-5)


def test_mean_soc_uses_full_session_including_rest():
    from Constrained_BO.hybrid_degradation import session_throughput_metrics

    i = np.concatenate([np.full(100, -2.0), np.zeros(100)])
    q = 7560.0
    soc = np.clip(0.2 + np.cumsum(-i) / q, 0.0, 1.0)
    session = {
        "current_a": i,
        "temperature_c": np.full(i.size, 25.0),
        "soc": soc,
        "q_rated_as": q,
        "initial_state": {"soc": 0.2},
    }
    m = session_throughput_metrics(session)
    assert np.isclose(m["mean_soc"], float(soc.mean()))
    assert np.isclose(m["duration_h"], i.size / 3600.0)
