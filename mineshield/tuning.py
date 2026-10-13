"""AI-threshold tuning on a dedicated tuning set (never the test seeds).

Run with ``python -m mineshield.tuning``. For each calibration quantile (threshold taken from the
normal-only validation seeds) and each AI persistence, the hybrid and Isolation-Forest heads are run
on tuning episodes (seeds 2100+, events placed exactly like the test suite) and event-level metrics
are printed. The chosen values are then written to config/default.json by hand and documented.
"""

from __future__ import annotations

import copy
import sys

import numpy as np

from .config import load_config
from .engine import HYBRID, get_model
from .evaluation import (_alerts_of, _events_of, _sim_kwargs, match_episode, n_event_slots, place_events,
                         run_episode, summarize_matches)
from .model import build_training_matrix
from .simulator import FaceSimulator

TUNING_SEED_START = 2100
TUNING_EPISODES = 8
QUANTILES = (0.995, 0.999, 0.9995)
PERSIST_S = (3, 5, 10)


def main() -> int:
    cfg = load_config()
    model = get_model(cfg)
    raw_val = model.raw_scores(build_training_matrix(cfg, cfg["model"]["validation_seeds"]))
    ep_s = float(cfg["evaluation"]["episode_s"])
    n_slots = n_event_slots(cfg, ep_s)
    hours = TUNING_EPISODES * ep_s / 3600.0
    print(f"tuning set: seeds {TUNING_SEED_START}-{TUNING_SEED_START + TUNING_EPISODES - 1}, "
          f"{TUNING_EPISODES} x {ep_s:g} s; validation rows {raw_val.size}")
    print(f"{'quantile':>9} {'persist':>7} | {'method':<18} {'prec':>6} {'recall':>6} {'f1':>6} {'FA/h':>6} {'delay':>6}")
    for q in QUANTILES:
        m = copy.copy(model)
        m.threshold = float(np.quantile(raw_val, q))
        for p in PERSIST_S:
            c = copy.deepcopy(cfg)
            c["model"]["persist_s"] = p
            matches = {HYBRID: [], "isolation_forest": []}
            counter = 0
            for k in range(TUNING_EPISODES):
                seed = TUNING_SEED_START + k
                sim = FaceSimulator(c, seed, **_sim_kwargs(c))
                counter = place_events(sim, seed, n_slots, counter)
                pipe = run_episode(sim, c, m, int(ep_s), modes=[HYBRID, "isolation_forest"])
                for mode in matches:
                    matches[mode].append(match_episode(_alerts_of(pipe, mode), _events_of(sim),
                                                       float(c["evaluation"]["match_grace_s"])))
            for mode, mm in matches.items():
                s = summarize_matches(mm, hours)
                print(f"{q:>9} {p:>7} | {mode:<18} {s['precision'] or 0:>6.3f} {s['recall'] or 0:>6.3f} "
                      f"{s['f1'] or 0:>6.3f} {s['fp_per_hour']:>6.2f} {s['median_delay_s'] or 0:>6.1f}")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
