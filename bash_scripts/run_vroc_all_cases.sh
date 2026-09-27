#!/usr/bin/env bash
set -euo pipefail

for case in {1..10}; do
  uv run --extra vroc --with-editable . scripts/build_vroc_vector_fields.py \
    --output-folder dvfs/vroc \
    --reference-phase 5 \
    --direction forward \
    --case "${case}"
done

for case in {1..10}; do
  uv run --extra vroc --with-editable . scripts/build_vroc_vector_fields.py \
    --output-folder dvfs/vroc \
    --reference-phase 5 \
    --direction backward \
    --case "${case}"
done
