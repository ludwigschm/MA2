"""
Neon Experiment Logger
======================

Purpose
-------
This script connects to the nearest reachable Pupil Labs Neon device, starts
recording, forwards experiment events as annotations, and stops the recording
at the end. Events become visible as Markers/Annotations after the recording is
uploaded to Pupil Cloud.

Dependencies
------------
Python 3.10+

Install the required packages with:

    pip install pupil-labs-realtime-api fastapi uvicorn

Usage Overview
--------------
Run the script from the command line, e.g.:

    python neon_experiment_logger.py --session "S01" --run "R001" --event-source stdin

When using ``--event-source stdin`` provide events in the following format (one
per line):

    trial_start trial=1 cond=A
    stim_onset trial=1 stim=A
    response trial=1 rt_ms=582 correct=true

When using ``--event-source http`` you can trigger events remotely:

    curl -X POST localhost:8081/event -H "Content-Type: application/json" \
         -d '{"name":"trial_start","data":{"trial":1,"cond":"A"}}'
    curl -X POST localhost:8081/control -H "Content-Type: application/json" -d '{"cmd":"stop"}'

The ``--simulate`` flag performs a dry run without contacting a real device,
which is useful for testing the integration with an experiment script.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from queue import Queue, Empty
from typing import Any, Dict, Iterable, Optional

try:
    from pupil_labs.realtime_api.simple import discover_one_device
except ImportError as exc:  # pragma: no cover - import guard for environments without the SDK
    discover_one_device = None  # type: ignore[assignment]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field, validator
    import uvicorn
except ImportError:  # pragma: no cover - optional dependency
    FastAPI = None  # type: ignore
    HTTPException = None  # type: ignore
    BaseModel = object  # type: ignore
    Field = lambda *a, **k: None  # type: ignore
    validator = lambda *a, **k: (lambda x: x)  # type: ignore
    uvicorn = None  # type: ignore


logger = logging.getLogger(__name__)


class ExitCodes:
    OK = 0
    DISCOVERY_ERROR = 2
    API_ERROR = 3
    EVENT_ERROR = 4


EVENT_NAME_PATTERN = r"^[a-zA-Z0-9_.:-]{1,64}$"
EVENT_KEY_PATTERN = r"^[a-zA-Z0-9_]{1,32}$"
RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY = 0.5


@dataclass
class RecorderState:
    device: Any | None
    recording_active: bool = False
    simulate: bool = False
    session: Optional[str] = None
    run: Optional[str] = None

    def recording_start(self, timeout: int) -> str:
        if self.recording_active:
            logger.info("Recording already active; ignoring additional start request.")
            return "already_running"
        if self.simulate:
            logger.info("[SIMULATION] Would start recording on Neon device.")
            self.recording_active = True
            return "simulated"
        if not self.device:
            raise RuntimeError("Device handle missing.")
        logger.debug("Calling device.recording_start() with timeout=%s", timeout)
        try:
            recording_id = self.device.recording_start(timeout=timeout)
        except TypeError:
            logger.debug("recording_start() does not accept timeout argument; retrying without it.")
            recording_id = self.device.recording_start()
        except Exception as exc:  # pragma: no cover - depends on SDK behavior
            logger.exception("Failed to start recording: %s", exc)
            raise
        else:
            logger.info("Recording started with ID: %s", recording_id)
            self.recording_active = True
            return recording_id

    def recording_stop_and_save(self, timeout: int) -> None:
        if not self.recording_active:
            logger.info("Recording not active; nothing to stop.")
            return
        if self.simulate:
            logger.info("[SIMULATION] Would stop & save recording on Neon device.")
            self.recording_active = False
            return
        if not self.device:
            raise RuntimeError("Device handle missing.")
        logger.debug("Calling device.recording_stop_and_save() with timeout=%s", timeout)
        try:
            self.device.recording_stop_and_save(timeout=timeout)
        except TypeError:
            logger.debug("recording_stop_and_save() does not accept timeout argument; retrying without it.")
            self.device.recording_stop_and_save()
        except Exception as exc:  # pragma: no cover
            logger.exception("Failed to stop and save recording: %s", exc)
            raise
        else:
            logger.info("Recording stopped and saved.")
            self.recording_active = False

    def close(self) -> None:
        if self.simulate:
            logger.info("[SIMULATION] Would close Neon device connection.")
            return
        if not self.device:
            return
        try:
            self.device.close()
        except Exception as exc:  # pragma: no cover
            logger.warning("Error while closing device: %s", exc)
        else:
            logger.info("Device connection closed.")


class NeonExperimentLogger:
    """Encapsulates connection management and event forwarding to Neon."""

    def __init__(
        self,
        session: Optional[str],
        run: Optional[str],
        timeout: int,
        simulate: bool,
    ) -> None:
        self.state = RecorderState(device=None, simulate=simulate, session=session, run=run)
        self.timeout = timeout
        self._shutdown_event = threading.Event()
        self._event_queue: "Queue[tuple[str, Dict[str, Any] | None]]" = Queue()

    # ------------------------------------------------------------------
    # Connection & lifecycle management
    # ------------------------------------------------------------------
    def connect_device(self) -> None:
        if self.state.simulate:
            logger.info("Running in simulation mode - no device discovery performed.")
            return
        if discover_one_device is None:
            logger.error(
                "pupil-labs-realtime-api is not installed. Please install it via 'pip install pupil-labs-realtime-api'."
            )
            if IMPORT_ERROR:
                raise IMPORT_ERROR
            raise RuntimeError("pupil-labs-realtime-api unavailable")
        logger.info("Discovering Neon device (timeout %ss)...", self.timeout)
        device = None
        try:
            device = discover_one_device(timeout=self.timeout)
        except TypeError:
            logger.debug("discover_one_device() does not accept timeout parameter; using fallback loop.")
            device = self._discover_device_with_timeout()
        except Exception as exc:
            logger.error("Device discovery failed: %s", exc)
            raise
        if device is None:
            raise RuntimeError("No Neon device discovered.")
        logger.info("Connected to device: %s", getattr(device, "device_id", "unknown"))
        self.state.device = device

    def _discover_device_with_timeout(self) -> Any:
        if discover_one_device is None:
            raise RuntimeError("Discovery function unavailable")
        deadline = time.monotonic() + self.timeout
        last_error: Optional[Exception] = None
        while time.monotonic() < deadline and not self._shutdown_event.is_set():
            try:
                device = discover_one_device()
            except Exception as exc:  # pragma: no cover - depends on SDK behavior
                last_error = exc
                logger.debug("Discovery retry due to error: %s", exc)
                time.sleep(0.5)
                continue
            if device:
                return device
            time.sleep(0.5)
        if last_error:
            raise last_error
        raise RuntimeError("Discovery timed out after %s seconds" % self.timeout)

    def start_recording(self) -> str:
        return self.state.recording_start(timeout=self.timeout)

    def stop_recording(self) -> None:
        self.state.recording_stop_and_save(timeout=self.timeout)

    def close_device(self) -> None:
        self.state.close()

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------
    def send_event(self, name: str, data: Optional[Dict[str, Any]]) -> None:
        name = name.strip()
        if not self.state.recording_active:
            logger.warning("Recording is not active; event '%s' skipped.", name)
            return
        payload = self._prepare_payload(data)
        logger.debug("Prepared payload for '%s': %s", name, payload)
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                if self.state.simulate:
                    logger.info("[SIMULATION] Event '%s' with data=%s", name, payload)
                else:
                    if not self.state.device:
                        raise RuntimeError("Device handle missing.")
                    try:
                        self.state.device.recording_event(name=name, data=payload, timeout=self.timeout)
                    except TypeError:
                        logger.debug("recording_event() does not accept timeout argument; retrying without it.")
                        self.state.device.recording_event(name=name, data=payload)
                logger.info("Event sent: %s", name)
                break
            except Exception as exc:  # pragma: no cover - depends on SDK behavior
                if attempt >= RETRY_ATTEMPTS:
                    logger.error("Failed to send event '%s' after %s attempts: %s", name, attempt, exc)
                    raise
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Transient error while sending event '%s' (attempt %s/%s): %s. Retrying in %.1fs",
                    name,
                    attempt,
                    RETRY_ATTEMPTS,
                    exc,
                    delay,
                )
                time.sleep(delay)

    def enqueue_event(self, name: str, data: Optional[Dict[str, Any]]) -> None:
        self._event_queue.put((name, data))

    def process_event_queue(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                name, data = self._event_queue.get(timeout=0.5)
            except Empty:
                continue
            try:
                self.send_event(name, data)
            except Exception:
                logger.exception("Error while sending event '%s'", name)
            finally:
                self._event_queue.task_done()

    def _prepare_payload(self, data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if data:
            payload.update(data)
        payload["ts_local"] = datetime.now(timezone.utc).astimezone().isoformat()
        if self.state.session is not None:
            payload.setdefault("session", self.state.session)
        if self.state.run is not None:
            payload.setdefault("run", self.state.run)
        return payload

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def request_shutdown(self) -> None:
        self._shutdown_event.set()

    def wait_for_queue(self) -> None:
        self._event_queue.join()


# ----------------------------------------------------------------------
# Event source helpers
# ----------------------------------------------------------------------

def parse_event_line(line: str) -> tuple[str, Dict[str, Any] | None]:
    line = line.strip()
    if not line:
        raise ValueError("Empty line")
    tokens = line.split()
    name = tokens[0]
    validate_event_name(name)
    data: Dict[str, Any] = {}
    for token in tokens[1:]:
        if "=" not in token:
            raise ValueError(f"Malformed key=value pair: '{token}'")
        key, value = token.split("=", 1)
        validate_event_key(key)
        data[key] = parse_value(value)
    return name, data or None


def parse_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if lowered.startswith("0x"):
            return int(lowered, 16)
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def validate_event_name(name: str) -> None:
    import re

    if not re.fullmatch(EVENT_NAME_PATTERN, name):
        raise ValueError(f"Invalid event name '{name}'.")


def validate_event_key(key: str) -> None:
    import re

    if not re.fullmatch(EVENT_KEY_PATTERN, key):
        raise ValueError(f"Invalid key '{key}'.")


# ----------------------------------------------------------------------
# HTTP API definitions
# ----------------------------------------------------------------------


if isinstance(BaseModel, type):

    class EventPayload(BaseModel):
        name: str = Field(..., regex=EVENT_NAME_PATTERN, max_length=64)
        data: Optional[Dict[str, Any]] = None

        @validator("data", pre=True, always=True)
        def ensure_dict(cls, value: Any) -> Optional[Dict[str, Any]]:  # pragma: no cover - simple validation
            if value is None:
                return None
            if not isinstance(value, dict):
                raise ValueError("data must be an object")
            return value

    class ControlPayload(BaseModel):
        cmd: str

        @validator("cmd")
        def validate_cmd(cls, value: str) -> str:  # pragma: no cover
            if value not in {"start", "stop"}:
                raise ValueError("cmd must be 'start' or 'stop'")
            return value

else:
    EventPayload = ControlPayload = None  # type: ignore


class HTTPEventServer:
    def __init__(self, logger_instance: NeonExperimentLogger, host: str, port: int) -> None:
        if FastAPI is None or uvicorn is None:
            raise RuntimeError("FastAPI/uvicorn are not installed.")
        self.logger_instance = logger_instance
        self.host = host
        self.port = port
        self._thread: Optional[threading.Thread] = None
        self._server: Optional[uvicorn.Server] = None
        self.app = self._create_app()

    def _create_app(self) -> FastAPI:
        app = FastAPI(title="Neon Experiment Logger API", version="1.0")
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"]
            ,
            allow_headers=["*"],
        )

        @app.post("/event")
        async def post_event(payload: EventPayload) -> Dict[str, Any]:
            logger.debug("HTTP event received: %s", payload)
            try:
                data = normalize_dict(payload.data)
            except ValueError as exc:
                logger.warning("Invalid event payload data: %s", exc)
                raise HTTPException(status_code=400, detail=str(exc))
            try:
                self.logger_instance.enqueue_event(payload.name, data)
            except ValueError as exc:
                logger.warning("Invalid event payload: %s", exc)
                raise HTTPException(status_code=400, detail=str(exc))
            return {"status": "queued"}

        @app.post("/control")
        async def post_control(payload: ControlPayload) -> Dict[str, Any]:
            logger.info("HTTP control command: %s", payload.cmd)
            if payload.cmd == "start":
                try:
                    recording_id = self.logger_instance.start_recording()
                except Exception as exc:
                    logger.error("Failed to start recording via control endpoint: %s", exc)
                    raise HTTPException(status_code=500, detail=str(exc))
                return {"status": "recording", "id": recording_id}
            if payload.cmd == "stop":
                try:
                    self.logger_instance.stop_recording()
                except Exception as exc:
                    logger.error("Failed to stop recording via control endpoint: %s", exc)
                    raise HTTPException(status_code=500, detail=str(exc))
                return {"status": "stopped"}
            raise HTTPException(status_code=400, detail="Unsupported command")

        return app

    def start(self) -> None:
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="info")
        self._server = uvicorn.Server(config=config)
        self._thread = threading.Thread(target=self._server.run, name="uvicorn-server", daemon=True)
        self._thread.start()
        logger.info("HTTP server running on %s:%s", self.host, self.port)

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)
            logger.info("HTTP server stopped.")


def normalize_dict(data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if data is None:
        return None
    normalized: Dict[str, Any] = {}
    for key, value in data.items():
        validate_event_key(key)
        normalized[key] = value
    return normalized


# ----------------------------------------------------------------------
# Event source runners
# ----------------------------------------------------------------------

def run_stdin_source(logger_instance: NeonExperimentLogger) -> None:
    logger.info("Reading events from STDIN. Press Ctrl+D to finish.")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            name, data = parse_event_line(line)
        except ValueError as exc:
            logger.error("Invalid event line: %s", exc)
            raise
        logger_instance.enqueue_event(name, data)
    logger.info("STDIN source ended.")


def run_demo_source(logger_instance: NeonExperimentLogger) -> None:
    logger.info("Starting demo event generator.")
    trials = 5
    for trial in range(1, trials + 1):
        if logger_instance._shutdown_event.is_set():
            break
        logger_instance.enqueue_event("trial_start", {"trial": trial, "cond": "A"})
        time.sleep(1)
        logger_instance.enqueue_event("stim_onset", {"trial": trial, "stim": f"stim_{trial}"})
        time.sleep(0.5)
        logger_instance.enqueue_event("response", {"trial": trial, "rt_ms": 500 + trial * 10, "correct": trial % 2 == 0})
        time.sleep(1)
    logger.info("Demo event generator finished.")


def run_http_source(logger_instance: NeonExperimentLogger, port: int) -> HTTPEventServer:
    server = HTTPEventServer(logger_instance, host="0.0.0.0", port=port)
    server.start()
    return server


# ----------------------------------------------------------------------
# CLI and main program
# ----------------------------------------------------------------------


def configure_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream experiment events to a Pupil Labs Neon device.")
    parser.add_argument("--session", type=str, default=None, help="Session identifier to include in events.")
    parser.add_argument("--run", type=str, default=None, help="Run/block identifier to include in events.")
    parser.add_argument("--event-source", choices=["stdin", "demo", "http"], default="stdin", help="Source of experiment events.")
    parser.add_argument("--port", type=int, default=8081, help="Port for HTTP server when --event-source=http.")
    parser.add_argument("--timeout", type=int, default=10, help="Timeout in seconds for discovery and API calls.")
    parser.add_argument("--simulate", type=int, default=0, help="Dry run without contacting a device.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


class GracefulShutdown:
    def __init__(self, logger_instance: NeonExperimentLogger):
        self.logger_instance = logger_instance
        self.received_signal = False

    def __call__(self, signum: int, frame: Any) -> None:  # pragma: no cover - signal handling side effect
        if self.received_signal:
            logger.warning("Multiple termination signals received; forcing exit soon.")
        else:
            logger.info("Received signal %s - shutting down gracefully.", signum)
        self.received_signal = True
        self.logger_instance.request_shutdown()


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.debug)
    simulate = bool(args.simulate)
    neon_logger = NeonExperimentLogger(session=args.session, run=args.run, timeout=args.timeout, simulate=simulate)

    shutdown_handler = GracefulShutdown(neon_logger)
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        neon_logger.connect_device()
    except Exception as exc:
        logger.error("Could not connect to Neon device: %s", exc)
        return ExitCodes.DISCOVERY_ERROR

    try:
        neon_logger.start_recording()
    except Exception as exc:
        logger.error("Failed to start recording: %s", exc)
        neon_logger.close_device()
        return ExitCodes.API_ERROR

    event_thread = threading.Thread(target=neon_logger.process_event_queue, name="event-worker", daemon=True)
    event_thread.start()

    http_server: Optional[HTTPEventServer] = None
    try:
        if args.event_source == "stdin":
            run_stdin_source(neon_logger)
        elif args.event_source == "demo":
            run_demo_source(neon_logger)
            neon_logger.request_shutdown()
        elif args.event_source == "http":
            try:
                http_server = run_http_source(neon_logger, args.port)
            except Exception as exc:
                logger.error("Failed to start HTTP server: %s", exc)
                raise
            while not neon_logger._shutdown_event.is_set():
                time.sleep(0.5)
        else:
            raise ValueError(f"Unsupported event source: {args.event_source}")
    except ValueError as exc:
        logger.error("Event error: %s", exc)
        neon_logger.request_shutdown()
        neon_logger.stop_recording()
        neon_logger.close_device()
        return ExitCodes.EVENT_ERROR
    except Exception as exc:
        logger.exception("Unhandled error while processing events: %s", exc)
        neon_logger.request_shutdown()
        neon_logger.stop_recording()
        neon_logger.close_device()
        return ExitCodes.API_ERROR
    finally:
        if http_server:
            http_server.stop()
        neon_logger.request_shutdown()
        neon_logger.wait_for_queue()
        event_thread.join(timeout=2)
        neon_logger.stop_recording()
        neon_logger.close_device()

    return ExitCodes.OK


if __name__ == "__main__":
    sys.exit(main())
