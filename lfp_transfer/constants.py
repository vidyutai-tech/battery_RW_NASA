"""LFP dataset comment filters (mirrors NASA ``STEP_MODE_COMMENTS``)."""

from __future__ import annotations

from typing import Dict, Optional, Set

# Step-level comment is taken from the first sample in each step array.
LFP_STEP_MODE_COMMENTS: Dict[str, Optional[Set[str]]] = {
  # 553 / 660 steps — operational random-walk protocol
  "rw_only": {"Random Walking"},
  # Random walk + periodic reference characterisation
  "rw_plus_reference": {"Random Walking", "1C Reference"},
  # All protocol types (includes HPPC pulses)
  "all": None,
}

DEFAULT_CELL_ID = "LFP"
