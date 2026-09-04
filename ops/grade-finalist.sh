#!/usr/bin/env bash
# Verify, isolate-convert, and privately score one finalist checkpoint.
set -euo pipefail
umask 077

readonly AUDIT_SCHEMA_VERSION="finalist-grading-audit-v2"
readonly SCORE_VERSION="unlearning-v2"
readonly PINNED_TEST_DATASET_REVISION="f7938fad4be1b9559433adf6f3edfab6088750ba003371de7c7505b5da05353b"
readonly MAX_CONVERTED_BYTES=536870912

usage() {
    cat >&2 <<'EOF'
Usage:
  ops/grade-finalist.sh \
    --image unlearning-grader:test-v2 \
    --checkpoint /organizer/submissions/<submission-id>.pt \
    --expected-sha256 <leaderboard-artifact-sha256> \
    --submission-id <submission-id> \
    --grading-root /grading-data/assets \
    --report /organizer/reports/<submission-id>.json \
    [--gpu 0]
EOF
    exit 2
}

IMAGE=
CHECKPOINT=
EXPECTED_SHA256=
SUBMISSION_ID=
GRADING_ROOT=
REPORT=
GPU=0
IMAGE_RESOLVE_TIMEOUT_SECONDS="${FINALIST_IMAGE_RESOLVE_TIMEOUT_SECONDS:-30}"
VERIFIER_TIMEOUT_SECONDS="${FINALIST_VERIFIER_TIMEOUT_SECONDS:-300}"
CONVERSION_TIMEOUT_SECONDS="${FINALIST_CONVERSION_TIMEOUT_SECONDS:-360}"
SCORING_TIMEOUT_SECONDS="${FINALIST_SCORING_TIMEOUT_SECONDS:-1800}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --image) IMAGE="${2:-}"; shift 2 ;;
        --checkpoint) CHECKPOINT="${2:-}"; shift 2 ;;
        --expected-sha256) EXPECTED_SHA256="${2:-}"; shift 2 ;;
        --submission-id) SUBMISSION_ID="${2:-}"; shift 2 ;;
        --grading-root) GRADING_ROOT="${2:-}"; shift 2 ;;
        --report) REPORT="${2:-}"; shift 2 ;;
        --gpu) GPU="${2:-}"; shift 2 ;;
        *) usage ;;
    esac
done

[ -n "${IMAGE}" ] && [ -n "${CHECKPOINT}" ] && [ -n "${EXPECTED_SHA256}" ] || usage
[ -n "${SUBMISSION_ID}" ] && [ -n "${GRADING_ROOT}" ] && [ -n "${REPORT}" ] || usage
case "${EXPECTED_SHA256}" in
    *[!0-9a-f]*|'') echo "expected SHA-256 must be lowercase hexadecimal" >&2; exit 2 ;;
esac
[ "${#EXPECTED_SHA256}" -eq 64 ] || { echo "expected SHA-256 must have 64 characters" >&2; exit 2; }
if [[ ! "${SUBMISSION_ID}" =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]]; then
    echo "submission ID must be a canonical lowercase UUID" >&2
    exit 2
fi
case "${GPU}" in *[!0-9]*|'') echo "GPU must be a numeric device ID" >&2; exit 2 ;; esac
for timeout_setting in \
    "image resolve:${IMAGE_RESOLVE_TIMEOUT_SECONDS}" \
    "verifier:${VERIFIER_TIMEOUT_SECONDS}" \
    "conversion:${CONVERSION_TIMEOUT_SECONDS}" \
    "scoring:${SCORING_TIMEOUT_SECONDS}"; do
    timeout_name="${timeout_setting%%:*}"
    timeout_value="${timeout_setting#*:}"
    case "${timeout_value}" in
        *[!0-9]*|'') echo "${timeout_name} timeout must be a positive integer" >&2; exit 2 ;;
    esac
    [ "${timeout_value}" -gt 0 ] || { echo "${timeout_name} timeout must be positive" >&2; exit 2; }
done
case "${CHECKPOINT}" in /*) ;; *) echo "checkpoint path must be absolute" >&2; exit 2 ;; esac
case "${GRADING_ROOT}" in /*) ;; *) echo "grading root must be absolute" >&2; exit 2 ;; esac
case "${REPORT}" in /*) ;; *) echo "report path must be absolute" >&2; exit 2 ;; esac
[ "$(basename -- "${CHECKPOINT}")" = "${SUBMISSION_ID}.pt" ] || {
    echo "checkpoint basename must be exactly ${SUBMISSION_ID}.pt" >&2
    exit 2
}
if [[ "${REPORT}" == *.json ]]; then
    AUDIT_REPORT="${REPORT%.json}.audit.json"
else
    AUDIT_REPORT="${REPORT}.audit.json"
fi
[ -f "${CHECKPOINT}" ] && [ ! -L "${CHECKPOINT}" ] || { echo "checkpoint must be a regular non-symlink file" >&2; exit 1; }
[ -d "${GRADING_ROOT}" ] && [ ! -L "${GRADING_ROOT}" ] || { echo "grading root must be a real directory" >&2; exit 1; }
[ ! -e "${REPORT}" ] && [ ! -L "${REPORT}" ] || { echo "refusing to overwrite report: ${REPORT}" >&2; exit 1; }
[ ! -e "${AUDIT_REPORT}" ] && [ ! -L "${AUDIT_REPORT}" ] || { echo "refusing to overwrite audit report: ${AUDIT_REPORT}" >&2; exit 1; }
if [ -n "$(find "${GRADING_ROOT}" -perm /077 -print -quit)" ]; then
    echo "grading root must be owner-only; group/other permissions are forbidden" >&2
    exit 1
fi

for required in \
    splits/test_split.pt \
    score_cache/refs.pt \
    score_cache/M_o__test.npz \
    imagenet_test; do
    [ -e "${GRADING_ROOT}/${required}" ] || { echo "private test asset is missing: ${required}" >&2; exit 1; }
done

actual_sha256="$(sha256sum -- "${CHECKPOINT}" | cut -d ' ' -f 1)"
[ "${actual_sha256}" = "${EXPECTED_SHA256}" ] || {
    echo "checkpoint SHA-256 does not match the leaderboard" >&2
    exit 1
}

# A mutable tag is accepted only as an operator convenience. Resolve it once,
# validate Docker's content-addressed identifier, and use only that identifier
# for every subsequent container so a concurrent retag cannot switch code.
if ! IMAGE_ID="$(
    timeout --foreground --signal=TERM --kill-after=5s \
        "${IMAGE_RESOLVE_TIMEOUT_SECONDS}s" \
        docker image inspect --format '{{.Id}}' -- "${IMAGE}"
)"; then
    echo "could not resolve grader image to a local immutable image ID" >&2
    exit 1
fi
[ "${#IMAGE_ID}" -eq 71 ] && [ "${IMAGE_ID#sha256:}" != "${IMAGE_ID}" ] || {
    echo "Docker returned an invalid grader image ID" >&2
    exit 1
}
image_digest="${IMAGE_ID#sha256:}"
case "${image_digest}" in
    *[!0-9a-f]*|'') echo "Docker returned an invalid grader image ID" >&2; exit 1 ;;
esac

run_root="$(mktemp -d -t hackathon-finalist.XXXXXXXX)"
audit_tmp=
report_publish_tmp=
report_published=0
audit_published=0
publication_complete=0
cleanup() {
    local cid_file container_cid
    for cid_file in \
        "${run_root}/verifier-before.cid" \
        "${run_root}/converter.cid" \
        "${run_root}/scorer.cid" \
        "${run_root}/verifier-after.cid"; do
        if [ -s "${cid_file}" ]; then
            read -r container_cid < "${cid_file}" || true
            case "${container_cid:-}" in
                *[!0-9a-f]*|'') ;;
                *)
                    if [ "${#container_cid}" -eq 64 ]; then
                        timeout --signal=KILL 15s \
                            docker rm -f "${container_cid}" >/dev/null 2>&1 || true
                    fi
                    ;;
            esac
        fi
    done
    if [ "${publication_complete}" != 1 ]; then
        if [ "${audit_published}" = 1 ] && [ -n "${audit_tmp}" ] && \
                [ -e "${AUDIT_REPORT}" ] && [ "${AUDIT_REPORT}" -ef "${audit_tmp}" ]; then
            rm -f -- "${AUDIT_REPORT}"
        fi
        if [ "${report_published}" = 1 ] && [ -n "${report_publish_tmp}" ] && \
                [ -e "${REPORT}" ] && [ "${REPORT}" -ef "${report_publish_tmp}" ]; then
            rm -f -- "${REPORT}"
        fi
    fi
    if [ -n "${audit_tmp}" ]; then
        rm -f -- "${audit_tmp}"
    fi
    if [ -n "${report_publish_tmp}" ]; then
        rm -f -- "${report_publish_tmp}"
    fi
    rm -rf -- "${run_root}"
}
trap cleanup EXIT INT TERM
mkdir -m 0700 \
    "${run_root}/input" \
    "${run_root}/trusted" \
    "${run_root}/score-output"
cp --reflink=auto -- "${CHECKPOINT}" "${run_root}/input/submission.pt"
chmod 0400 "${run_root}/input/submission.pt"
[ "$(sha256sum -- "${run_root}/input/submission.pt" | cut -d ' ' -f 1)" = "${EXPECTED_SHA256}" ] || {
    echo "staged checkpoint SHA-256 mismatch" >&2
    exit 1
}

container_uid="$(id -u)"
container_gid="$(id -g)"
if [ "${container_uid}" = 0 ]; then
    container_uid=65534
    container_gid=65534
    chmod 0711 "${run_root}"
    chown -R 65534:65534 "${run_root}/input"
    chmod 0700 "${run_root}/input"
    chmod 0400 "${run_root}/input/submission.pt"
fi

# Trusted bundle verification happens before any executable torch metadata is
# parsed. The verifier pins the manifest, split, refs, M_o, cache, and complete
# image-content tree to the generated v2 release.
verify_bundle() {
    local cid_file="$1"
    timeout --foreground --signal=TERM --kill-after=10s \
        "${VERIFIER_TIMEOUT_SECONDS}s" docker run --rm \
        --cidfile "${cid_file}" \
        --network none \
        --read-only \
        --cap-drop ALL \
        --security-opt no-new-privileges:true \
        --user "$(id -u):$(id -g)" \
        --mount "type=bind,src=${GRADING_ROOT},dst=/grading-data/assets,readonly" \
        --tmpfs /tmp:rw,noexec,nosuid,nodev,size=64m \
        --entrypoint python \
        "${IMAGE_ID}" /app/verify_test_bundle.py \
        --grading-root /grading-data/assets \
        --verify-only
}

if ! verify_bundle "${run_root}/verifier-before.cid"; then
    echo "private test bundle verification failed" >&2
    exit 1
fi

# Untrusted phase: the vulnerable .pt parser receives no network, credential,
# GPU, private test tree, or root privilege. Only its safetensors output crosses
# into the trusted phase. Safetensors bytes travel over stdout into a host-side
# limiter; the container has no host output mount, and both the stream and its
# only writable tmpfs are independently bounded.
untrusted_safe="${run_root}/untrusted.safetensors"
set +e
timeout --foreground --signal=TERM --kill-after=10s \
    "${CONVERSION_TIMEOUT_SECONDS}s" docker run --rm \
    --cidfile "${run_root}/converter.cid" \
    --log-driver none \
    --network none \
    --read-only \
    --cap-drop ALL \
    --security-opt no-new-privileges:true \
    --pids-limit 64 \
    --memory 4g \
    --cpus 2 \
    --user "${container_uid}:${container_gid}" \
    --mount "type=bind,src=${run_root}/input,dst=/input,readonly" \
    --tmpfs /tmp:rw,noexec,nosuid,nodev,size=1g \
    --entrypoint python \
    "${IMAGE_ID}" /app/convert_checkpoint.py \
    --input /input/submission.pt \
    --output-fd 1 \
    --max-bytes "${MAX_CONVERTED_BYTES}" \
    2>/dev/null | head -c "$((MAX_CONVERTED_BYTES + 1))" > "${untrusted_safe}"
converter_pipeline_status=("${PIPESTATUS[@]}")
set -e
converted_size="$(stat -c %s -- "${untrusted_safe}")"
if [ "${converted_size}" -gt "${MAX_CONVERTED_BYTES}" ]; then
    echo "isolated checkpoint conversion exceeded the output byte limit" >&2
    exit 1
fi
if [ "${converter_pipeline_status[0]}" -ne 0 ] || \
        [ "${converter_pipeline_status[1]}" -ne 0 ]; then
    echo "isolated checkpoint conversion failed or timed out" >&2
    exit 1
fi
[ "${converted_size}" -gt 0 ] && [ -f "${untrusted_safe}" ] && [ ! -L "${untrusted_safe}" ] || {
    echo "converter did not produce a regular safetensors file" >&2
    exit 1
}
safe_checkpoint="${run_root}/trusted/submission.safetensors"
cp --reflink=auto --no-preserve=ownership,mode -- \
    "${untrusted_safe}" "${safe_checkpoint}"
chmod 0400 "${safe_checkpoint}"
[ -f "${safe_checkpoint}" ] && [ ! -L "${safe_checkpoint}" ] || {
    echo "could not restage converted checkpoint into the trusted directory" >&2
    exit 1
}
safe_sha256="$(sha256sum -- "${safe_checkpoint}" | cut -d ' ' -f 1)"

report_dir="$(dirname -- "${REPORT}")"
staged_report="${run_root}/score-output/report.json"

# Trusted phase: only non-executable safetensors plus the private test bundle
# are visible. The original participant .pt is not mounted, and the report is
# held under the private run directory until the bundle passes a second exact
# verification.
if ! timeout --foreground --signal=TERM --kill-after=10s \
    "${SCORING_TIMEOUT_SECONDS}s" docker run --rm \
    --cidfile "${run_root}/scorer.cid" \
    --network none \
    --read-only \
    --cap-drop ALL \
    --security-opt no-new-privileges:true \
    --gpus "device=${GPU}" \
    --shm-size 8g \
    --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=${safe_checkpoint},dst=/input/submission.safetensors,readonly" \
    --mount "type=bind,src=${GRADING_ROOT},dst=/grading-data/assets,readonly" \
    --mount "type=bind,src=${run_root}/score-output,dst=/output" \
    "${IMAGE_ID}" score \
    --phase test \
    --split /grading-data/assets/splits/test_split.pt \
    --refs /grading-data/assets/score_cache/refs.pt \
    --image-root /grading-data/assets/imagenet_test \
    --ckpt /input/submission.safetensors \
    --mo-cache /grading-data/assets/score_cache/M_o__test.npz \
    --tag "${SUBMISSION_ID}" \
    --report /output/report.json \
    --device cuda:0; then
    echo "private test scoring failed or timed out" >&2
    exit 1
fi

# Detect any persistent private-data mutation that occurred after the first
# verification or while the scorer was reading it. Nothing has been published
# to the canonical report directory at this point.
if ! verify_bundle "${run_root}/verifier-after.cid"; then
    echo "private test bundle changed during scoring; discarding result" >&2
    exit 1
fi

[ -f "${staged_report}" ] && [ ! -L "${staged_report}" ] || {
    echo "scorer did not produce a regular non-symlink report" >&2
    exit 1
}
[ "$(sha256sum -- "${safe_checkpoint}" | cut -d ' ' -f 1)" = "${safe_sha256}" ] || {
    echo "converted checkpoint changed while it was being scored" >&2
    exit 1
}

# Refuse to attest a report produced by an image with a different scoring
# contract. These four values are generated by the trusted scorer, not copied
# from participant-controlled metadata.
python3 - \
    "${staged_report}" \
    "${SUBMISSION_ID}" \
    "${SCORE_VERSION}" \
    "${PINNED_TEST_DATASET_REVISION}" <<'PY'
import json
import sys

report_path, submission_id, score_version, dataset_revision = sys.argv[1:]
with open(report_path, "r", encoding="utf-8") as stream:
    report = json.load(stream)
if not isinstance(report, dict):
    raise SystemExit("scorer report must be a JSON object")
expected = {
    "phase": "test",
    "tag": submission_id,
    "score_version": score_version,
    "dataset_revision": dataset_revision,
}
for key, value in expected.items():
    if report.get(key) != value:
        raise SystemExit(
            f"scorer report {key!r} does not match the finalist grading pin"
        )
PY

report_sha256="$(sha256sum -- "${staged_report}" | cut -d ' ' -f 1)"

# Stage both immutable files in the destination directory, then publish each
# with an atomic no-clobber hard link. Cleanup rolls back the first link if the
# second cannot be installed, so a failed run leaves neither canonical file.
mkdir -p -- "${report_dir}"
[ -d "${report_dir}" ] && [ ! -L "${report_dir}" ] || {
    echo "report parent must be a real directory" >&2
    exit 1
}
[ ! -e "${AUDIT_REPORT}" ] && [ ! -L "${AUDIT_REPORT}" ] || {
    echo "refusing to overwrite audit report: ${AUDIT_REPORT}" >&2
    exit 1
}
[ ! -e "${REPORT}" ] && [ ! -L "${REPORT}" ] || {
    echo "refusing to overwrite report: ${REPORT}" >&2
    exit 1
}
report_publish_tmp="$(mktemp "${report_dir}/.finalist-report.XXXXXXXX")"
cp --reflink=auto --no-preserve=ownership,mode -- \
    "${staged_report}" "${report_publish_tmp}"
chmod 0444 "${report_publish_tmp}"
[ "$(sha256sum -- "${report_publish_tmp}" | cut -d ' ' -f 1)" = "${report_sha256}" ] || {
    echo "staged score report SHA-256 mismatch" >&2
    exit 1
}
audit_tmp="$(mktemp "${report_dir}/.finalist-audit.XXXXXXXX")"
chmod 0600 "${audit_tmp}"
printf '%s\n' \
    '{' \
    "  \"schema_version\": \"${AUDIT_SCHEMA_VERSION}\"," \
    "  \"submission_id\": \"${SUBMISSION_ID}\"," \
    "  \"original_checkpoint_sha256\": \"${actual_sha256}\"," \
    "  \"converted_safetensors_sha256\": \"${safe_sha256}\"," \
    "  \"final_report_sha256\": \"${report_sha256}\"," \
    "  \"score_version\": \"${SCORE_VERSION}\"," \
    "  \"test_dataset_revision\": \"${PINNED_TEST_DATASET_REVISION}\"," \
    "  \"grader_image_id\": \"${IMAGE_ID}\"" \
    '}' > "${audit_tmp}"
chmod 0444 "${audit_tmp}"
report_published=1
if ! ln -- "${report_publish_tmp}" "${REPORT}"; then
    echo "refusing to overwrite report: ${REPORT}" >&2
    exit 1
fi
audit_published=1
if ! ln -- "${audit_tmp}" "${AUDIT_REPORT}"; then
    echo "refusing to overwrite audit report: ${AUDIT_REPORT}" >&2
    exit 1
fi

# Flush both link targets and the containing directory before declaring the
# two-file publication complete.
python3 - "${REPORT}" "${AUDIT_REPORT}" "${report_dir}" <<'PY'
import os
import sys

report_path, audit_path, report_dir = sys.argv[1:]
for path in (report_path, audit_path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
directory_descriptor = os.open(
    report_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
)
try:
    os.fsync(directory_descriptor)
finally:
    os.close(directory_descriptor)
PY
publication_complete=1
rm -f -- "${audit_tmp}" "${report_publish_tmp}"
audit_tmp=
report_publish_tmp=

echo "wrote ${REPORT}"
echo "wrote ${AUDIT_REPORT}"
