from __future__ import annotations

import json
import secrets
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


DATABASE_SCHEMA_VERSION = 2
_UNSET = object()


class LostClaimError(RuntimeError):
    """Raised when a worker tries to finish a claim it no longer owns."""


class SubmissionLimitError(RuntimeError):
    """Raised without advancing the counter when a team's limit is exhausted."""

    def __init__(self, team_name: str, max_count: int):
        self.team_name = team_name
        self.max_count = max_count
        super().__init__(
            f"{team_name} already reached its submission limit ({max_count})"
        )


SCHEMA = """
CREATE TABLE IF NOT EXISTS team_counters (
    team_name TEXT PRIMARY KEY,
    next_submission_number INTEGER NOT NULL CHECK (next_submission_number >= 1)
);

CREATE TABLE IF NOT EXISTS submissions (
    id TEXT PRIMARY KEY,
    team_name TEXT NOT NULL,
    team_number INTEGER NOT NULL,
    submission_number INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    source_relative_path TEXT NOT NULL,
    source_size_bytes INTEGER NOT NULL,
    source_mtime_ns INTEGER NOT NULL,
    source_ctime_ns INTEGER NOT NULL,
    source_sha256 TEXT NOT NULL,
    artifact_relative_path TEXT NOT NULL UNIQUE,
    receipt_relative_path TEXT NOT NULL UNIQUE,
    marker_relative_path TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'done', 'error')),
    attempt_count INTEGER NOT NULL DEFAULT 0,
    worker_id TEXT,
    worker_gpu INTEGER,
    claim_token TEXT,
    captured_at TEXT NOT NULL,
    queued_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    error TEXT,
    report_relative_path TEXT,
    audit_relative_path TEXT,
    score_depth TEXT,
    acc_f REAL,
    acc_r REAL,
    cka_f_o REAL,
    cka_r_o REAL,
    cka_per_depth_json TEXT,
    aus REAL,
    rus_o REAL,
    final_score REAL,
    f1 REAL,
    report_json TEXT,
    UNIQUE (team_name, submission_number),
    UNIQUE (team_name, source_sha256)
);

CREATE INDEX IF NOT EXISTS submissions_status_queue
ON submissions(status, queued_at, team_number, submission_number);

CREATE INDEX IF NOT EXISTS submissions_source_version
ON submissions(team_name, source_relative_path, source_size_bytes,
               source_mtime_ns, source_ctime_ns);

CREATE TABLE IF NOT EXISTS source_versions (
    team_name TEXT NOT NULL,
    source_relative_path TEXT NOT NULL,
    source_size_bytes INTEGER NOT NULL,
    source_mtime_ns INTEGER NOT NULL,
    source_ctime_ns INTEGER NOT NULL,
    source_sha256 TEXT NOT NULL,
    disposition TEXT NOT NULL CHECK (
        disposition IN ('queued', 'duplicate', 'limit_exceeded')
    ),
    submission_id TEXT NOT NULL,
    PRIMARY KEY (
        team_name, source_relative_path, source_size_bytes,
        source_mtime_ns, source_ctime_ns
    )
);
"""


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path.parent.chmod(0o700)
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            connection.execute("BEGIN IMMEDIATE")
            submission_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(submissions)")
            }
            added_claim_token = "claim_token" not in submission_columns
            if added_claim_token:
                connection.execute(
                    "ALTER TABLE submissions ADD COLUMN claim_token TEXT"
                )
                # A legacy running row has no owner credential and therefore can
                # never be completed safely after this migration.
                connection.execute(
                    "UPDATE submissions SET status = 'queued', worker_id = NULL, "
                    "worker_gpu = NULL, started_at = NULL, finished_at = NULL, "
                    "error = 'requeued during claim-token migration' "
                    "WHERE status = 'running'"
                )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS submissions_claim_token "
                "ON submissions(claim_token) WHERE claim_token IS NOT NULL"
            )
            source_schema_row = connection.execute(
                "SELECT sql FROM sqlite_master "
                "WHERE type = 'table' AND name = 'source_versions'"
            ).fetchone()
            source_schema = source_schema_row["sql"] if source_schema_row else ""
            if "limit_exceeded" not in source_schema:
                connection.execute(
                    "ALTER TABLE source_versions RENAME TO source_versions_legacy"
                )
                connection.execute(
                    """
                    CREATE TABLE source_versions (
                        team_name TEXT NOT NULL,
                        source_relative_path TEXT NOT NULL,
                        source_size_bytes INTEGER NOT NULL,
                        source_mtime_ns INTEGER NOT NULL,
                        source_ctime_ns INTEGER NOT NULL,
                        source_sha256 TEXT NOT NULL,
                        disposition TEXT NOT NULL CHECK (
                            disposition IN ('queued', 'duplicate', 'limit_exceeded')
                        ),
                        submission_id TEXT NOT NULL,
                        PRIMARY KEY (
                            team_name, source_relative_path, source_size_bytes,
                            source_mtime_ns, source_ctime_ns
                        )
                    )
                    """
                )
                connection.execute(
                    "INSERT INTO source_versions SELECT * FROM source_versions_legacy"
                )
                connection.execute("DROP TABLE source_versions_legacy")
            connection.execute(f"PRAGMA user_version={DATABASE_SCHEMA_VERSION}")
        self.path.chmod(0o600)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def allocate_submission_number(
        self, team_name: str, max_count: int | None = None
    ) -> int:
        if max_count is not None and (
            isinstance(max_count, bool)
            or not isinstance(max_count, int)
            or max_count < 0
        ):
            raise ValueError("max_count must be a non-negative integer or None")
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT next_submission_number FROM team_counters WHERE team_name = ?",
                (team_name,),
            ).fetchone()
            if row is None:
                number = 1
                if max_count is not None and number > max_count:
                    raise SubmissionLimitError(team_name, max_count)
                connection.execute(
                    "INSERT INTO team_counters(team_name, next_submission_number) "
                    "VALUES (?, ?)",
                    (team_name, 2),
                )
            else:
                number = int(row[0])
                if max_count is not None and number > max_count:
                    raise SubmissionLimitError(team_name, max_count)
                connection.execute(
                    "UPDATE team_counters SET next_submission_number = ? "
                    "WHERE team_name = ?",
                    (number + 1, team_name),
                )
            return number

    def has_source_version(
        self,
        team_name: str,
        source_relative_path: str,
        size_bytes: int,
        mtime_ns: int,
        ctime_ns: int,
    ) -> bool:
        with self.connect() as connection:
            return (
                connection.execute(
                    "SELECT 1 FROM source_versions WHERE team_name = ? "
                    "AND source_relative_path = ? AND source_size_bytes = ? "
                    "AND source_mtime_ns = ? AND source_ctime_ns = ? LIMIT 1",
                    (
                        team_name,
                        source_relative_path,
                        size_bytes,
                        mtime_ns,
                        ctime_ns,
                    ),
                ).fetchone()
                is not None
            )

    def find_team_sha(self, team_name: str, sha256: str) -> dict | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM submissions WHERE team_name = ? AND source_sha256 = ?",
                (team_name, sha256),
            ).fetchone()
            return dict(row) if row else None

    def submission_exists(self, submission_id: str) -> bool:
        with self.connect() as connection:
            return (
                connection.execute(
                    "SELECT 1 FROM submissions WHERE id = ? LIMIT 1",
                    (submission_id,),
                ).fetchone()
                is not None
            )

    def record_source_version(
        self,
        *,
        team_name: str,
        source_relative_path: str,
        size_bytes: int,
        mtime_ns: int,
        ctime_ns: int,
        sha256: str,
        disposition: str,
        submission_id: str,
    ) -> None:
        if disposition not in {"queued", "duplicate", "limit_exceeded"}:
            raise ValueError("invalid source-version disposition")
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO source_versions(
                    team_name, source_relative_path, source_size_bytes,
                    source_mtime_ns, source_ctime_ns, source_sha256,
                    disposition, submission_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    team_name,
                    source_relative_path,
                    size_bytes,
                    mtime_ns,
                    ctime_ns,
                    sha256,
                    disposition,
                    submission_id,
                ),
            )

    def ingest_marker(self, marker: dict) -> bool:
        required = {
            "submission_id",
            "team_name",
            "team_number",
            "submission_number",
            "model_name",
            "source_relative_path",
            "size_bytes",
            "source_mtime_ns",
            "source_ctime_ns",
            "sha256",
            "artifact_relative_path",
            "receipt_relative_path",
            "marker_relative_path",
            "captured_at",
            "ready_at",
        }
        missing = required.difference(marker)
        if missing:
            raise ValueError(f"ready marker is missing fields: {sorted(missing)}")
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM submissions WHERE id = ?",
                (marker["submission_id"],),
            ).fetchone()
            if existing is not None:
                comparisons = {
                    "team_name": marker["team_name"],
                    "team_number": marker["team_number"],
                    "submission_number": marker["submission_number"],
                    "model_name": marker["model_name"],
                    "source_relative_path": marker["source_relative_path"],
                    "source_size_bytes": marker["size_bytes"],
                    "source_mtime_ns": marker["source_mtime_ns"],
                    "source_ctime_ns": marker["source_ctime_ns"],
                    "source_sha256": marker["sha256"],
                    "artifact_relative_path": marker["artifact_relative_path"],
                    "receipt_relative_path": marker["receipt_relative_path"],
                    "marker_relative_path": marker["marker_relative_path"],
                    "captured_at": marker["captured_at"],
                    "queued_at": marker["ready_at"],
                }
                if any(existing[field] != value for field, value in comparisons.items()):
                    raise ValueError("ready marker conflicts with its existing submission UUID")
                return False
            else:
                try:
                    connection.execute(
                        """
                        INSERT INTO submissions(
                            id, team_name, team_number, submission_number, model_name,
                            source_relative_path, source_size_bytes, source_mtime_ns,
                            source_ctime_ns, source_sha256, artifact_relative_path,
                            receipt_relative_path, marker_relative_path, status,
                            captured_at, queued_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?)
                        """,
                        (
                            marker["submission_id"],
                            marker["team_name"],
                            marker["team_number"],
                            marker["submission_number"],
                            marker["model_name"],
                            marker["source_relative_path"],
                            marker["size_bytes"],
                            marker["source_mtime_ns"],
                            marker["source_ctime_ns"],
                            marker["sha256"],
                            marker["artifact_relative_path"],
                            marker["receipt_relative_path"],
                            marker["marker_relative_path"],
                            marker["captured_at"],
                            marker["ready_at"],
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ValueError(
                        "ready marker conflicts with another submission identity"
                    ) from exc
                inserted = True
            connection.execute(
                "INSERT INTO team_counters(team_name, next_submission_number) VALUES (?, ?) "
                "ON CONFLICT(team_name) DO UPDATE SET next_submission_number = "
                "MAX(next_submission_number, excluded.next_submission_number)",
                (marker["team_name"], int(marker["submission_number"]) + 1),
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO source_versions(
                    team_name, source_relative_path, source_size_bytes,
                    source_mtime_ns, source_ctime_ns, source_sha256,
                    disposition, submission_id
                ) VALUES (?, ?, ?, ?, ?, ?, 'queued', ?)
                """,
                (
                    marker["team_name"],
                    marker["source_relative_path"],
                    marker["size_bytes"],
                    marker["source_mtime_ns"],
                    marker["source_ctime_ns"],
                    marker["sha256"],
                    marker["submission_id"],
                ),
            )
            return inserted

    def claim_next(self, gpu: int, worker_id: str, started_at: str) -> dict | None:
        claim_token = secrets.token_hex(32)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM submissions WHERE status = 'queued' "
                "ORDER BY queued_at, team_number, submission_number LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            updated = connection.execute(
                "UPDATE submissions SET status = 'running', worker_id = ?, "
                "worker_gpu = ?, claim_token = ?, started_at = ?, finished_at = NULL, "
                "attempt_count = attempt_count + 1, error = NULL "
                "WHERE id = ? AND status = 'queued' AND claim_token IS NULL",
                (worker_id, gpu, claim_token, started_at, row["id"]),
            )
            if updated.rowcount != 1:
                return None
            claimed = connection.execute(
                "SELECT * FROM submissions WHERE id = ?", (row["id"],)
            ).fetchone()
            return dict(claimed)

    def requeue_running(self) -> int:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            claims = connection.execute(
                "SELECT id, claim_token FROM submissions WHERE status = 'running'"
            ).fetchall()
            requeued = 0
            for claim in claims:
                result = connection.execute(
                    "UPDATE submissions SET status = 'queued', worker_id = NULL, "
                    "worker_gpu = NULL, claim_token = NULL, started_at = NULL, "
                    "finished_at = NULL, error = 'requeued after grader restart' "
                    "WHERE id = ? AND status = 'running' AND claim_token IS ?",
                    (claim["id"], claim["claim_token"]),
                )
                requeued += result.rowcount
            return requeued

    def requeue_claim(self, submission_id: str, claim_token: str) -> bool:
        """Requeue exactly one owned claim, invalidating its token atomically."""
        with self.connect() as connection:
            result = connection.execute(
                "UPDATE submissions SET status = 'queued', worker_id = NULL, "
                "worker_gpu = NULL, claim_token = NULL, started_at = NULL, "
                "finished_at = NULL, error = 'claim explicitly requeued' "
                "WHERE id = ? AND status = 'running' AND claim_token = ?",
                (submission_id, claim_token),
            )
            return result.rowcount == 1

    def mark_done(
        self,
        submission_id: str,
        claim_token: str,
        finished_at: str,
        metrics: dict,
        report_relative_path: str,
        audit_relative_path: str,
    ) -> None:
        with self.connect() as connection:
            result = connection.execute(
                """
                UPDATE submissions SET status = 'done', finished_at = ?, error = NULL,
                    report_relative_path = ?, audit_relative_path = ?,
                    score_depth = ?, acc_f = ?, acc_r = ?, cka_f_o = ?, cka_r_o = ?,
                    cka_per_depth_json = ?, aus = ?, rus_o = ?, final_score = ?,
                    f1 = ?, report_json = ?
                WHERE id = ? AND status = 'running' AND claim_token = ?
                """,
                (
                    finished_at,
                    report_relative_path,
                    audit_relative_path,
                    metrics["score_depth"],
                    metrics["acc_f"],
                    metrics["acc_r"],
                    metrics["cka_f_o"],
                    metrics["cka_r_o"],
                    metrics["cka_per_depth_json"],
                    metrics["aus"],
                    metrics["rus_o"],
                    metrics["final_score"],
                    metrics["f1_alias"],
                    metrics["report_json"],
                    submission_id,
                    claim_token,
                ),
            )
            if result.rowcount != 1:
                raise LostClaimError(
                    "submission claim was lost before publishing score"
                )

    def mark_error(
        self, submission_id: str, claim_token: str, finished_at: str, error: str
    ) -> None:
        with self.connect() as connection:
            result = connection.execute(
                "UPDATE submissions SET status = 'error', finished_at = ?, "
                "error = ? WHERE id = ? AND status = 'running' AND claim_token = ?",
                (finished_at, error[-4000:], submission_id, claim_token),
            )
            if result.rowcount != 1:
                raise LostClaimError(
                    "submission claim was lost before recording error"
                )

    def retry(
        self,
        submission_id: str,
        claim_token: str | None | object = _UNSET,
    ) -> bool:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT claim_token FROM submissions "
                "WHERE id = ? AND status = 'error'",
                (submission_id,),
            ).fetchone()
            if row is None:
                return False
            expected_token = (
                row["claim_token"] if claim_token is _UNSET else claim_token
            )
            result = connection.execute(
                "UPDATE submissions SET status = 'queued', worker_id = NULL, "
                "worker_gpu = NULL, claim_token = NULL, started_at = NULL, "
                "finished_at = NULL, error = NULL, report_relative_path = NULL, "
                "audit_relative_path = NULL, score_depth = NULL, acc_f = NULL, "
                "acc_r = NULL, cka_f_o = NULL, cka_r_o = NULL, "
                "cka_per_depth_json = NULL, aus = NULL, rus_o = NULL, "
                "final_score = NULL, f1 = NULL, report_json = NULL "
                "WHERE id = ? AND status = 'error' AND claim_token IS ?",
                (submission_id, expected_token),
            )
            return result.rowcount == 1

    def rows(self) -> list[dict]:
        with self.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM submissions ORDER BY team_number, submission_number"
                )
            ]

    def summary(self) -> dict:
        with self.connect() as connection:
            counts = {
                row["status"]: row["count"]
                for row in connection.execute(
                    "SELECT status, COUNT(*) AS count FROM submissions GROUP BY status"
                )
            }
            rows = [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM submissions ORDER BY team_number, submission_number"
                )
            ]
        return {"counts": counts, "submissions": rows}

    @staticmethod
    def row_for_json(row: dict) -> dict:
        fields = (
            "id",
            "team_name",
            "team_number",
            "submission_number",
            "model_name",
            "source_relative_path",
            "source_size_bytes",
            "source_sha256",
            "status",
            "attempt_count",
            "worker_gpu",
            "captured_at",
            "queued_at",
            "started_at",
            "finished_at",
            "error",
            "score_depth",
            "acc_f",
            "acc_r",
            "cka_f_o",
            "cka_r_o",
            "aus",
            "rus_o",
            "final_score",
            "f1",
            "report_relative_path",
            "audit_relative_path",
        )
        return {field: row.get(field) for field in fields}
