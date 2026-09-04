#!/usr/bin/env bash
set -euo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
readonly BOOTSTRAP_VENV="${PROJECT_ROOT}/.bootstrap-venv"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/.env"
    set +a
fi

python3 -m venv "${BOOTSTRAP_VENV}"
"${BOOTSTRAP_VENV}/bin/python" -m pip install --disable-pip-version-check \
    -r "${SCRIPT_DIR}/bootstrap-requirements.txt"
"${BOOTSTRAP_VENV}/bin/python" "${SCRIPT_DIR}/download-private-grading.py"

docker build \
    -f "${PROJECT_ROOT}/grading_docker/Dockerfile" \
    -t "${TA_GRADING_IMAGE:-hackathon/private-test-grader:2026.09}" \
    "${PROJECT_ROOT}"

"${SCRIPT_DIR}/check.sh"
printf '[bootstrap] ready; run %s\n' "${SCRIPT_DIR}/run.sh"

