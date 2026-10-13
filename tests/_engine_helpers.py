"""Shared helpers for the engine / API / evaluation tests.

``small_model()`` trains a deliberately small Isolation Forest once per test session (1 train seed
x 900 s, 1 validation seed x 600 s) and caches it; nothing is written to data/.
"""

from __future__ import annotations

import json
import math
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mineshield.config import load_config  # noqa: E402
from mineshield.model import AnomalyModel, build_training_matrix  # noqa: E402

SUMMARY_KEYS = {"id", "opened_t", "last_t", "active", "category", "title", "shields", "channels",
                "severity", "priority", "rules", "ai_score_max"}
DETAIL_KEYS = SUMMARY_KEYS | {"explanation", "evidence", "features", "ai_detail", "data_quality", "neighbors",
                              "verification_steps", "uncertainty", "priority_formula", "series", "window",
                              "matched_event"}
EVENT_KEYS = {"id", "type", "label", "category", "onset_t", "end_t", "shields", "channels", "magnitude",
              "magnitude_unit", "active"}
STATE_KEYS = {"sim", "counts", "shields", "heatmap", "series", "alerts", "events", "devices"}
SHIELD_KEYS = {"id", "index", "gateway", "state", "time_in_state_s", "legs", "mean_bar", "force_MN",
               "deviation_bar", "ai_score", "status"}
CSV_HEADER = "t_s,timestamp,seq,device_id,shield_id,channel_id,value,unit,quality,state,gt_label,gt_event_id"


def cfg():
    return load_config()


@lru_cache(maxsize=1)
def small_model() -> AnomalyModel:
    c = load_config()
    m = AnomalyModel(c).fit(build_training_matrix(c, [1000], 900))
    m.calibrate(build_training_matrix(c, [2000], 600))
    m.meta.update({"trained_on": "normal-only synthetic seed 1000 (small test model)",
                   "threshold_note": "score 1.0 = 99.5th percentile of validation normal data"})
    return m


def _no_nan(x):
    raise ValueError(f"non-finite JSON constant {x}")


def strict_json(obj) -> str:
    """Serialise and re-parse; fails on NaN/inf anywhere."""
    s = json.dumps(obj, allow_nan=False)
    json.loads(s, parse_constant=_no_nan)
    return s


def strict_loads(text: str):
    return json.loads(text, parse_constant=_no_nan)


def finite_or_none(x) -> bool:
    return x is None or (isinstance(x, (int, float)) and math.isfinite(x))
