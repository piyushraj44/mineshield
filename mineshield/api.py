"""FastAPI application (docs/API_CONTRACT.md). Serves the API under /api and the frontend at /.

``create_app()`` builds an app; the module-level ``app`` uses the default config and the cached /
freshly trained model (``engine.get_model``), created in the lifespan handler. Endpoints that touch
the engine are plain ``def`` functions (run in the threadpool); the engine serialises access with
its own lock. Errors are returned as HTTP 422 / 404 / 409 with ``{"detail": "<readable string>"}``.
"""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt

from . import __version__
from .config import PROJECT_ROOT, load_config
from .engine import WINDOW_DEFAULT_S, LiveEngine, get_model, json_clean
from .evaluation import RESULTS_PATH, load_results, run_evaluation, save_results
from .model import AnomalyModel
from .physics import piston_area_m2
from .simulator import EVENT_TYPES
from .telemetry import TelemetryValidator

FRONTEND_DIR = PROJECT_ROOT / "frontend"
CSV_CHUNK_LINES = 2000
NO_RESULTS = {"available": False, "message": "Run: python -m mineshield.evaluation"}


# --------------------------------------------------------------------------- request models
# Strict numeric types: JSON booleans and numeric strings are rejected (422) instead of being coerced.
class ResetBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: StrictInt | None = None
    n_shields: StrictInt | None = None
    legs_per_shield: StrictInt | None = None
    noise_bar: StrictFloat | None = None
    load_trend_bar_per_h: StrictFloat | None = None


class StepBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    steps: StrictInt = Field(1, ge=1, le=60)
    shield: str | None = None
    window_s: StrictFloat = WINDOW_DEFAULT_S


class InjectBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str
    shield: str
    leg: StrictInt | None = None
    magnitude: StrictFloat | None = None
    duration_s: StrictFloat | None = None


class IngestBody(BaseModel):
    readings: list[Any]


# --------------------------------------------------------------------------- helpers
def _validation_detail(exc: RequestValidationError) -> str:
    parts = []
    for err in exc.errors():
        if err.get("type") == "json_invalid":
            parts.append("request body is not valid JSON")
            continue
        loc = [str(x) for x in err.get("loc", ()) if x not in ("body", "query", "path")]
        msg = str(err.get("msg", "invalid value"))
        if not loc and tuple(err.get("loc", ())) == ("body",):
            loc = ["request body"]
        parts.append(f"{'.'.join(loc)}: {msg}" if loc else msg)
    return "; ".join(dict.fromkeys(parts)) or "invalid request"


def build_config_payload(cfg: dict) -> dict[str, Any]:
    """Contract GET /api/config."""
    return {
        "version": cfg.get("version", __version__),
        "units": cfg["units"],
        "limits": cfg["limits"],
        "piston_area_m2": round(float(piston_area_m2(float(cfg["piston_bore_m"]))), 4),
        "defaults": cfg["defaults"],
        "ranges": cfg["ranges"],
        "speeds": cfg["speeds"],
        "poll_ms": cfg["poll_ms"],
        "event_types": [dict(e) for e in EVENT_TYPES],
        "disclaimer": cfg["disclaimer"],
    }


def _chunked(lines: Iterator[str], n: int = CSV_CHUNK_LINES) -> Iterator[str]:
    buf: list[str] = []
    for line in lines:
        buf.append(line)
        if len(buf) >= n:
            yield "".join(buf)
            buf = []
    if buf:
        yield "".join(buf)


# --------------------------------------------------------------------------- app factory
def create_app(cfg: dict | None = None, model: AnomalyModel | None = None, *,
               model_path: str | Path | None = None, results_path: str | Path | None = None,
               frontend_dir: str | Path | None = None,
               evaluation_options: dict[str, Any] | None = None) -> FastAPI:
    """Build the app. ``model`` (if given) is used instead of ``get_model(cfg, model_path)``;
    ``evaluation_options`` are extra keyword arguments for ``run_evaluation`` (tests)."""
    cfg = cfg if cfg is not None else load_config()
    res_path = Path(results_path) if results_path is not None else RESULTS_PATH
    fe_dir = Path(frontend_dir) if frontend_dir is not None else FRONTEND_DIR
    eval_opts = dict(evaluation_options or {})
    config_payload = build_config_payload(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        m = model if model is not None else get_model(cfg, model_path)
        engine = LiveEngine(cfg, m)
        engine.reset(seed=cfg["defaults"]["seed"])
        app.state.model = m
        app.state.engine = engine
        yield

    app = FastAPI(title="MineShield AI", version=__version__, lifespan=lifespan,
                  description="Research prototype: simulated powered-support leg pressures and "
                              "abnormal-pattern detection. Not a certified safety system.")
    app.state.cfg = cfg
    app.state.eval_lock = threading.Lock()
    app.state.results_path = res_path

    @app.middleware("http")
    async def _no_stale_assets(request: Request, call_next):
        # Frontend files must revalidate (ETag) so an updated app.js is never mixed with an old one.
        response = await call_next(request)
        if not request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.exception_handler(RequestValidationError)
    async def _on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": _validation_detail(exc)})

    def engine_of(request: Request) -> LiveEngine:
        return request.app.state.engine

    # ---------------------------------------------------------------- config & simulation
    @app.get("/api/config")
    def get_config() -> JSONResponse:
        return JSONResponse(config_payload)

    @app.post("/api/sim/reset")
    def sim_reset(body: ResetBody, request: Request) -> JSONResponse:
        params = body.model_dump(exclude_none=True)
        try:
            info = engine_of(request).reset(**params)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        return JSONResponse({"ok": True, "sim": info})

    @app.post("/api/sim/step")
    def sim_step(body: StepBody, request: Request) -> JSONResponse:
        try:
            st = engine_of(request).step_state(body.steps, body.shield, body.window_s)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        return JSONResponse(st)

    @app.post("/api/sim/inject")
    def sim_inject(body: InjectBody, request: Request) -> JSONResponse:
        try:
            ev = engine_of(request).inject(body.type, body.shield, leg=body.leg, magnitude=body.magnitude,
                                           duration_s=body.duration_s)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        return JSONResponse(json_clean(ev))

    @app.get("/api/state")
    def get_state(request: Request, shield: str = "S01", window_s: float = WINDOW_DEFAULT_S) -> JSONResponse:
        return JSONResponse(engine_of(request).state(shield, window_s))

    @app.get("/api/alerts/{alert_id}")
    def get_alert(alert_id: str, request: Request) -> JSONResponse:
        try:
            d = engine_of(request).alert_detail(alert_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown alert id {alert_id!r}") from None
        return JSONResponse(d)

    # ---------------------------------------------------------------- evaluation
    @app.get("/api/evaluation")
    def get_evaluation() -> JSONResponse:
        data = load_results(res_path)
        return JSONResponse(data if data is not None else dict(NO_RESULTS))

    @app.post("/api/evaluation/run")
    def post_evaluation_run(request: Request) -> JSONResponse:
        lock: threading.Lock = request.app.state.eval_lock
        if not lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="An evaluation run is already in progress; "
                                                        "wait for it to finish.")
        try:
            res = run_evaluation(cfg, request.app.state.model, **eval_opts)
            save_results(res, res_path)
        finally:
            lock.release()
        return JSONResponse(res)

    # ---------------------------------------------------------------- exports
    @app.get("/api/export/experiment.json")
    def export_experiment(request: Request) -> JSONResponse:
        return JSONResponse(engine_of(request).export_experiment(),
                            headers={"Content-Disposition": 'attachment; filename="mineshield_experiment.json"'})

    @app.get("/api/export/telemetry.csv")
    def export_telemetry(request: Request) -> StreamingResponse:
        lines = engine_of(request).iter_csv()
        return StreamingResponse(_chunked(lines), media_type="text/csv; charset=utf-8",
                                 headers={"Content-Disposition": 'attachment; filename="mineshield_telemetry.csv"'})

    # ---------------------------------------------------------------- ingest
    @app.post("/api/ingest/validate")
    def ingest_validate(body: IngestBody, request: Request) -> JSONResponse:
        validator = TelemetryValidator(engine_of(request).channel_ids, cfg["limits"])
        accepted, rejected = validator.validate(body.readings)
        return JSONResponse(json_clean({"accepted": len(accepted), "rejected": rejected}))

    # ---------------------------------------------------------------- frontend (after /api routes)
    if fe_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(fe_dir), html=True), name="frontend")
    return app


app = create_app()
