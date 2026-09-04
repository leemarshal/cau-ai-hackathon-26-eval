from __future__ import annotations

import json
import math
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


DEFAULT_TIMEOUT_SECONDS = 5.0
DEFAULT_MAX_RESPONSE_BYTES = 64 * 1024


class ScorePostError(RuntimeError):
    """Raised when a score cannot be delivered to the configured endpoint."""


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Following a redirect could change POST to GET or send the score to a
        # different host. Returning None makes urllib surface the 3xx response
        # as HTTPError instead.
        return None


_OPENER = build_opener(_RejectRedirects())


def _validate_endpoint_url(endpoint_url: str) -> None:
    if not isinstance(endpoint_url, str) or not endpoint_url:
        raise ValueError("score endpoint URL must be a non-empty string")
    if any(character.isspace() for character in endpoint_url):
        raise ValueError("score endpoint URL must not contain whitespace")
    try:
        parsed = urlsplit(endpoint_url)
        hostname = parsed.hostname
        # Accessing port also rejects malformed and out-of-range port values.
        parsed.port
    except ValueError as exc:
        raise ValueError("score endpoint URL is invalid") from exc
    if parsed.scheme.lower() != "https" or not hostname:
        raise ValueError("score endpoint URL must be an HTTPS URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("score endpoint URL must not contain credentials")
    if parsed.fragment:
        raise ValueError("score endpoint URL must not contain a fragment")


def _validate_limits(timeout_seconds: float, max_response_bytes: int) -> None:
    try:
        finite_timeout = math.isfinite(timeout_seconds)
    except (TypeError, OverflowError):
        finite_timeout = False
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not finite_timeout
        or timeout_seconds <= 0
    ):
        raise ValueError("score POST timeout must be a positive finite number")
    if (
        isinstance(max_response_bytes, bool)
        or not isinstance(max_response_bytes, int)
        or max_response_bytes < 0
    ):
        raise ValueError("maximum response size must be a non-negative integer")


def _payload(team_id: int, score: float) -> bytes:
    if isinstance(team_id, bool) or not isinstance(team_id, int):
        raise TypeError("team_id must be an integer")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise TypeError("score must be a finite number")
    try:
        score_value = float(score)
    except OverflowError as exc:
        raise ValueError("score must be a finite number") from exc
    if not math.isfinite(score_value):
        raise ValueError("score must be a finite number")
    return json.dumps(
        {"team_id": team_id, "score": score_value},
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _consume_bounded(response, max_response_bytes: int) -> None:
    """Consume no more than the configured number of response bytes."""
    response.read(max_response_bytes)


def post_score(
    endpoint_url: str,
    team_id: int,
    score: float,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
) -> None:
    """POST one score as JSON, returning after an HTTPS 2xx status."""
    _validate_endpoint_url(endpoint_url)
    _validate_limits(timeout_seconds, max_response_bytes)
    body = _payload(team_id, score)
    request = Request(
        endpoint_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        try:
            response = _OPENER.open(request, timeout=float(timeout_seconds))
        except HTTPError as exc:
            with exc:
                _consume_bounded(exc, max_response_bytes)
            raise ScorePostError(
                f"score endpoint returned HTTP {exc.code}"
            ) from exc

        with response:
            status = response.getcode()
            if isinstance(status, int) and 200 <= status < 300:
                return
            _consume_bounded(response, max_response_bytes)
        if not isinstance(status, int) or not 200 <= status < 300:
            raise ScorePostError(
                f"score endpoint returned invalid HTTP status {status!r}"
            )
    except ScorePostError:
        raise
    except (URLError, TimeoutError, OSError, HTTPException) as exc:
        raise ScorePostError(f"score POST failed: {exc}") from exc
