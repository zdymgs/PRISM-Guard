#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'HELP'
Usage: bash competition/inference.sh <data_dir> <output_csv> [bundle]

Run the fixed challenge model on raw trials listed in <data_dir>/test.csv.
Supply the deployment checkpoint as [bundle] or PRISM_COMPETITION_BUNDLE.
Set PRISM_PYTHON to select a Python interpreter explicitly.
HELP
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  usage
  exit 0
fi
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  usage >&2
  exit 2
fi

DATA_DIR="$1"
OUTPUT_CSV="$2"
BUNDLE="${3:-${PRISM_COMPETITION_BUNDLE:-}}"
COMPETITION_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPOSITORY_ROOT="$(CDPATH= cd -- "$COMPETITION_ROOT/.." && pwd)"

if [ ! -d "$DATA_DIR" ] || [ ! -f "$DATA_DIR/test.csv" ]; then
  echo "Expected an existing raw-data directory containing test.csv: $DATA_DIR" >&2
  exit 2
fi
if [ -z "$BUNDLE" ] || [ ! -f "$BUNDLE" ]; then
  echo "Supply an existing challenge bundle using the third argument or PRISM_COMPETITION_BUNDLE." >&2
  exit 2
fi

if [ -n "${PRISM_PYTHON:-}" ]; then
  PYTHON_BIN="$PRISM_PYTHON"
elif [ -x "$REPOSITORY_ROOT/.venv/bin/python" ]; then
  PYTHON_BIN="$REPOSITORY_ROOT/.venv/bin/python"
elif [ -f "$REPOSITORY_ROOT/.venv/Scripts/python.exe" ]; then
  PYTHON_BIN="$REPOSITORY_ROOT/.venv/Scripts/python.exe"
else
  PYTHON_BIN="python"
fi

mkdir -p -- "$(dirname -- "$OUTPUT_CSV")"
TEMP_OUTPUT="$(mktemp -d)"
cleanup() {
  rm -rf -- "$TEMP_OUTPUT"
}
trap cleanup EXIT

"$PYTHON_BIN" "$COMPETITION_ROOT/student/runtime.py" \
  --bundle "$BUNDLE" \
  --data-root "$DATA_DIR" \
  --test-csv "$DATA_DIR/test.csv" \
  --output "$TEMP_OUTPUT"

cp -f -- "$TEMP_OUTPUT/submission.csv" "$OUTPUT_CSV"
for SIDECAR in diagnostics.json report.json; do
  if [ -f "$TEMP_OUTPUT/$SIDECAR" ]; then
    cp -f -- "$TEMP_OUTPUT/$SIDECAR" "$OUTPUT_CSV.$SIDECAR"
  fi
done
