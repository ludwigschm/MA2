"""HTTP client with retry, jittered backoff and circuit breaker support."""

from __future__ import annotations

import logging
import random
import socket
import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence
from urllib.parse import urljoin

try:  # pragma: no cover - optional dependency
    import requests
    from requests import Response, Session
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore[assignment]
    Response = object  # type: ignore[assignment]
    Session = object  # type: ignore[assignment]

from core.config import ApiClientSettings

__all__ = [
    "ApiError",
    "ApiNotFound",
    "ApiBadGateway",
    "ApiDnsError",
    "ApiTimeout",
    "CircuitOpenError",
    "HttpClient",
]


log = logging.getLogger(__name__)


class ApiError(RuntimeError):
    """Base class for structured API errors."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        error_code: str | None = None,
        endpoint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.error_code = error_code
        self.endpoint = endpoint


class ApiNotFound(ApiError):
    """Raised when the endpoint returns 404."""


class ApiBadGateway(ApiError):
    """Raised for server-side 5xx failures that warrant retries."""


class ApiDnsError(ApiError):
    """Raised when DNS resolution fails for the host."""


class ApiTimeout(ApiError):
    """Raised when the request times out."""


class CircuitOpenError(ApiError):
    """Raised when the circuit breaker refuses additional requests."""


def _is_dns_error(exc: BaseException) -> bool:
    current: Optional[BaseException] = exc  # type: ignore[assignment]
    while current:
        if isinstance(current, socket.gaierror):
            return True
        current = current.__cause__  # type: ignore[assignment]
    return False


@dataclass
class _CircuitBreaker:
    threshold: int
    reset_timeout: float

    def __post_init__(self) -> None:
        self._failures = 0
        self._state = "closed"
        self._opened_at = 0.0
        self._half_open_trial = False

    @property
    def state(self) -> str:
        return self._state

    def allow(self) -> bool:
        if self._state == "closed":
            return True
        now = time.monotonic()
        if self._state == "open":
            if now - self._opened_at >= self.reset_timeout:
                self._state = "half_open"
                self._half_open_trial = False
                return True
            return False
        if self._state == "half_open":
            if not self._half_open_trial:
                self._half_open_trial = True
                return True
            return False
        return True

    def record_success(self) -> None:
        self._failures = 0
        self._state = "closed"
        self._opened_at = 0.0
        self._half_open_trial = False

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= max(1, self.threshold):
            self._state = "open"
            self._opened_at = time.monotonic()


class HttpClient:
    """Wrapper around :mod:`requests` with health checks and retries."""

    def __init__(
        self,
        base_url: str,
        *,
        settings: ApiClientSettings,
        session_factory: Callable[[], Session] | None = None,
        name: str | None = None,
    ) -> None:
        if requests is None:  # pragma: no cover - optional dependency
            raise RuntimeError("requests is required for HttpClient")
        self.base_url = base_url.rstrip("/")
        self.settings = settings
        self._name = name or settings.name or "api"
        self._session_factory = session_factory or requests.Session  # type: ignore[assignment]
        self._session: Session = self._session_factory()
        self._circuit = _CircuitBreaker(
            threshold=max(1, settings.circuit_threshold),
            reset_timeout=max(1.0, settings.circuit_reset_s),
        )

    @property
    def circuit_state(self) -> str:
        return self._circuit.state

    def close(self) -> None:
        close_fn = getattr(self._session, "close", None)
        if callable(close_fn):
            close_fn()

    # ------------------------------------------------------------------
    def health_check(
        self,
        paths: Sequence[str],
        *,
        headers: Optional[dict[str, str]] = None,
    ) -> Optional[str]:
        last_error: Optional[ApiError] = None
        for path in paths:
            try:
                response = self.get(
                    path,
                    headers=headers,
                    allow_statuses={200, 204},
                    idempotent=True,
                )
            except ApiNotFound:
                continue
            except ApiError as exc:
                last_error = exc
                break
            else:
                if response.status_code in {200, 204}:
                    return path
        if last_error is not None:
            raise last_error
        return None

    def discover_path(
        self,
        method: str,
        paths: Sequence[str],
        *,
        json: Optional[dict] = None,
        headers: Optional[dict[str, str]] = None,
        allow_statuses: Optional[set[int]] = None,
        idempotent: bool = False,
    ) -> Optional[str]:
        for path in paths:
            try:
                response = self.request(
                    method,
                    path,
                    json=json,
                    headers=headers,
                    allow_statuses=allow_statuses,
                    idempotent=idempotent,
                )
            except ApiNotFound:
                continue
            except ApiError:
                continue
            else:
                status = getattr(response, "status_code", None)
                if allow_statuses is None or status in allow_statuses:
                    return path
        return None

    # ------------------------------------------------------------------
    def get(
        self,
        path: str,
        *,
        headers: Optional[dict[str, str]] = None,
        allow_statuses: Optional[set[int]] = None,
        idempotent: bool = True,
        timeout: Optional[float] = None,
    ) -> Response:
        return self.request(
            "GET",
            path,
            headers=headers,
            allow_statuses=allow_statuses,
            idempotent=idempotent,
            timeout=timeout,
        )

    def post(
        self,
        path: str,
        *,
        json: Optional[dict] = None,
        headers: Optional[dict[str, str]] = None,
        allow_statuses: Optional[set[int]] = None,
        idempotent: bool = False,
        timeout: Optional[float] = None,
    ) -> Response:
        return self.request(
            "POST",
            path,
            json=json,
            headers=headers,
            allow_statuses=allow_statuses,
            idempotent=idempotent,
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    def request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[dict] = None,
        headers: Optional[dict[str, str]] = None,
        allow_statuses: Optional[set[int]] = None,
        idempotent: bool = False,
        timeout: Optional[float] = None,
    ) -> Response:
        url = urljoin(self.base_url + "/", path.lstrip("/"))
        attempts = 1 + max(0, self.settings.retry_max if idempotent or method.upper() in {"GET", "HEAD"} else 0)
        delay = max(0.05, self.settings.backoff_factor)
        attempt = 0
        last_error: Optional[ApiError] = None
        while attempt < attempts:
            attempt += 1
            if not self._circuit.allow():
                raise CircuitOpenError(
                    f"circuit open for {self._name}",
                    endpoint=url,
                )
            start = time.perf_counter()
            try:
                response = self._session.request(
                    method,
                    url,
                    json=json,
                    headers=headers,
                    timeout=timeout or self.settings.timeout_s,
                )
            except requests.exceptions.Timeout as exc:  # type: ignore[union-attr]
                last_error = ApiTimeout(
                    f"timeout contacting {url}", endpoint=url
                )
                last_error.__cause__ = exc  # type: ignore[attr-defined]
                self._log_attempt(url, attempt, attempts, None, time.perf_counter() - start, error=last_error)
                self._circuit.record_failure()
            except requests.exceptions.RequestException as exc:  # type: ignore[union-attr]
                if _is_dns_error(exc):
                    last_error = ApiDnsError(f"dns failure contacting {url}", endpoint=url)
                else:
                    last_error = ApiBadGateway(f"connection error for {url}", endpoint=url)
                last_error.__cause__ = exc  # type: ignore[attr-defined]
                self._log_attempt(url, attempt, attempts, None, time.perf_counter() - start, error=last_error)
                self._circuit.record_failure()
            else:
                status = getattr(response, "status_code", None)
                if allow_statuses and status in allow_statuses:
                    self._log_attempt(url, attempt, attempts, status, time.perf_counter() - start)
                    self._circuit.record_success()
                    return response
                if 200 <= (status or 0) < 300:
                    self._log_attempt(url, attempt, attempts, status, time.perf_counter() - start)
                    self._circuit.record_success()
                    return response
                error: ApiError
                if status == 404:
                    error = ApiNotFound(f"{url} returned 404", status=status, endpoint=url)
                elif status in {502, 503, 504}:
                    error = ApiBadGateway(
                        f"{url} returned {status}", status=status, endpoint=url
                    )
                elif status in {408}:
                    error = ApiTimeout(f"{url} timed out", status=status, endpoint=url)
                else:
                    error = ApiError(f"{url} returned {status}", status=status, endpoint=url)
                self._log_attempt(url, attempt, attempts, status, time.perf_counter() - start, error=error)
                if status is not None and 400 <= status < 500 and status not in {408}:
                    self._circuit.record_failure()
                    raise error
                self._circuit.record_failure()
                last_error = error

            if attempt >= attempts or not self._should_retry(last_error):
                assert last_error is not None
                raise last_error
            sleep_time = self._compute_delay(delay, attempt)
            time.sleep(sleep_time)
        if last_error is None:
            raise ApiError(f"request to {url} failed")
        raise last_error

    # ------------------------------------------------------------------
    def _should_retry(self, error: Optional[ApiError]) -> bool:
        if error is None:
            return False
        if isinstance(error, ApiNotFound):
            return False
        return True

    def _compute_delay(self, base: float, attempt: int) -> float:
        exponent = max(0, attempt - 1)
        backoff = base * (2**exponent)
        jitter = random.uniform(0, backoff * 0.25)
        return min(backoff + jitter, 5.0)

    def _log_attempt(
        self,
        url: str,
        attempt: int,
        total: int,
        status: int | None,
        latency_s: float,
        *,
        error: Optional[ApiError] = None,
    ) -> None:
        extra = {
            "endpoint": url,
            "status": status,
            "attempt": attempt,
            "latency_ms": int(latency_s * 1000),
            "circuit_state": self._circuit.state,
        }
        if error is not None:
            extra["error_code"] = error.__class__.__name__
        if error is None:
            if attempt == 1:
                log.debug(
                    "%s request ok", self._name, extra=extra
                )
            else:
                log.info("%s request recovered", self._name, extra=extra)
            return
        if attempt >= total:
            log.warning("%s request failed", self._name, extra=extra)
        else:
            next_delay = self._compute_delay(self.settings.backoff_factor, attempt + 1)
            extra["retry_in_ms"] = int(next_delay * 1000)
            log.debug("%s request retry", self._name, extra=extra)

