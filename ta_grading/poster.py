from __future__ import annotations

import fcntl
import logging
import os
import socket
import stat
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from .config import Settings
from .database import Database
from .publish import publish_state
from .score_post import ScorePostError, post_score


LOGGER = logging.getLogger("ta-grader.poster")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _publish_post_state(settings: Settings, database: Database) -> None:
    try:
        publish_state(settings, database)
    except Exception:
        LOGGER.exception("score POST state changed but Admin publication failed")


@contextmanager
def _poster_lifetime_lock(settings: Settings):
    """Ensure recovery and delivery have exactly one live owner."""

    settings.state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if settings.state_root.is_symlink() or not settings.state_root.is_dir():
        raise RuntimeError("score poster state path is not a real directory")
    settings.state_root.chmod(0o700)
    lock_path = settings.state_root / "score-poster.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    lock = os.fdopen(descriptor, "a+b")
    try:
        metadata = os.fstat(lock.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("score poster lock is not a regular file")
        os.fchmod(lock.fileno(), 0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another score poster is already running") from exc
        yield
    finally:
        lock.close()


def deliver_one(settings: Settings, database: Database, poster_id: str) -> bool:
    """Deliver one due outbox item without changing its local grading result."""

    row = database.claim_next_score_post(poster_id, utc_now())
    if row is None:
        return False

    submission_id = row["submission_id"]
    claim_token = row["claim_token"]
    try:
        post_score(
            settings.score_post_url,
            int(row["team_id"]),
            float(row["score"]),
            timeout_seconds=settings.score_post_timeout_seconds,
        )
    except ScorePostError as exc:
        failed = datetime.now(timezone.utc)
        next_attempt = failed + timedelta(seconds=settings.score_post_retry_seconds)
        if not database.mark_score_post_failed(
            submission_id,
            claim_token,
            failed.isoformat(),
            str(exc),
            next_attempt.isoformat(),
        ):
            LOGGER.warning(
                "discarding POST failure from lost claim submission=%s",
                submission_id,
            )
            return True
        LOGGER.warning(
            "score POST failed; retry scheduled team_id=%d submission=%s error=%s",
            row["team_id"],
            submission_id,
            exc,
        )
        _publish_post_state(settings, database)
        return True

    delivered_at = utc_now()
    if not database.mark_score_post_delivered(
        submission_id, claim_token, delivered_at
    ):
        LOGGER.warning(
            "score endpoint accepted a lost POST claim submission=%s",
            submission_id,
        )
        return True

    LOGGER.info(
        "score POST delivered team_id=%d submission=%s score=%.10f",
        row["team_id"],
        submission_id,
        row["score"],
    )
    _publish_post_state(settings, database)
    return True


def poster_loop(settings: Settings, *, once: bool = False) -> bool:
    with _poster_lifetime_lock(settings):
        database = Database(settings.database_path)
        database.initialize()
        recovered = database.requeue_posting_score_posts()
        if recovered:
            LOGGER.warning("requeued %d interrupted score POSTs", recovered)
            _publish_post_state(settings, database)

        poster_id = f"{socket.gethostname()}:{os.getpid()}:score-poster"
        processed = False
        while True:
            delivered = deliver_one(settings, database, poster_id)
            processed = processed or delivered
            if once:
                return processed
            if not delivered:
                time.sleep(settings.worker_poll_seconds)
