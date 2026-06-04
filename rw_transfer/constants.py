"""Battery IDs and step-filter presets."""

from __future__ import annotations

CELL_ORDER = ("RW9", "RW10", "RW11", "RW12")

# Battery A–D mapping (source = RW9)
CELL_ALIASES = {
    "RW9": "A",
    "RW10": "B",
    "RW11": "C",
    "RW12": "D",
}

# Option A — RW operational (default)
RW_OPERATIONAL_COMMENTS = frozenset({
    "charge (random walk)",
    "discharge (random walk)",
    "rest (random walk)",
})

# Option B — RW + reference characterization
RW_PLUS_REFERENCE_COMMENTS = RW_OPERATIONAL_COMMENTS | frozenset({
    "reference charge",
    "reference discharge",
    "rest post reference charge",
    "rest post reference discharge",
    "rest prior low current discharge",
    "rest post low current discharge",
    "low current discharge at 0.04A",
})

STEP_MODE_COMMENTS = {
    "rw_operational": RW_OPERATIONAL_COMMENTS,
    "rw_plus_reference": RW_PLUS_REFERENCE_COMMENTS,
    "all": None,  # include every step
}

NASA_NOMINAL_Q_AS = 2.0 * 3600.0
