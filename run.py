"""Start MineShield AI: python run.py [--host 127.0.0.1] [--port 8000]

Ensures the anomaly model exists (trains it on the first run), then serves the API and the
frontend with uvicorn.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the MineShield AI research prototype (simulated data).")
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8000, help="port (default 8000)")
    args = ap.parse_args(argv)

    import uvicorn

    from mineshield.config import load_config
    from mineshield.engine import MODEL_PATH, get_model, model_is_current

    cfg = load_config()
    if not model_is_current(cfg):
        print(f"First run (or config changed): training the anomaly model on normal-only synthetic data, "
              f"saving to {MODEL_PATH} ...", flush=True)
        t0 = time.perf_counter()
        get_model(cfg)
        print(f"Model trained in {time.perf_counter() - t0:.1f} s.", flush=True)

    from mineshield.api import app

    print(f"MineShield AI running at http://{args.host}:{args.port}", flush=True)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
