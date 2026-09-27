#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="default"
RESP_METHOD="lung_volume"
RESP_CENTERING="ref"
REFERENCE_PHASE=5
DEVICE="cuda:0"
METHOD_NAME="vroc"
DVF_INPUT="dvfs/vroc"
LEAVE_OUT_PHASE=""

usage() {
    cat <<'EOF'
Usage: run_correspondence_all_cases.sh [options]

Runs scripts/build_correspondence_model.py for cases 1..10 and then aggregates
case-level metrics to cohort-level mean/std.

Options:
  --config NAME
  --resp-method {lung_volume}
  --resp-centering {ref|mean}
  --reference-phase N
  --device DEVICE
  --method-name NAME
  --dvf-input REL_PATH
  --leave-out-phase N
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --resp-method)
            RESP_METHOD="$2"
            shift 2
            ;;
        --resp-centering)
            RESP_CENTERING="$2"
            shift 2
            ;;
        --reference-phase)
            REFERENCE_PHASE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --method-name)
            METHOD_NAME="$2"
            shift 2
            ;;
        --dvf-input)
            DVF_INPUT="$2"
            shift 2
            ;;
        --leave-out-phase)
            LEAVE_OUT_PHASE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

for CASE in {1..10}; do
    echo "Running correspondence build for case ${CASE}"
    CMD=(
        uv run --with-editable . scripts/build_correspondence_model.py
        --case "${CASE}"
        --config "${CONFIG}"
        --resp-method "${RESP_METHOD}"
        --resp-centering "${RESP_CENTERING}"
        --reference-phase "${REFERENCE_PHASE}"
        --device "${DEVICE}"
        --direction both
        --method-name "${METHOD_NAME}"
        --dvf-root "${DVF_INPUT}"
    )

    if [[ -n "${LEAVE_OUT_PHASE}" ]]; then
        CMD+=(--leave-out-phase "${LEAVE_OUT_PHASE}")
    fi

    "${CMD[@]}"
done

echo "Aggregating results over 10 cases"
uv run --with-editable . scripts/aggregate_correspondence_cases.py \
    --config "${CONFIG}" \
    --resp-method "${RESP_METHOD}" \
    --resp-centering "${RESP_CENTERING}" \
    --reference-phase "${REFERENCE_PHASE}" \
    --method-name "${METHOD_NAME}"
