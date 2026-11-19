#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
WORKSPACE="$(cd "${RUN_DIR}/.." && pwd -P)"
if [[ -f "${WORKSPACE}/../huggingface/dataset/data/input_manifest.json" ]]; then
  RUN_DIR="$(cd "${WORKSPACE}/../huggingface/dataset" && pwd -P)"
fi
export V12_RUN_DIR="${DELIGHT_FM_RUN_DIR:-${V12_RUN_DIR:-${RUN_DIR}}}"
export DELIGHT_FM_RUN_DIR="$V12_RUN_DIR"
export DELIGHT_FM_GIT_ROOT="$WORKSPACE"
STAGE="${1:---help}"
case "$STAGE" in
  report|cluster)
    if command -v module >/dev/null 2>&1 && [[ -z "${REPORT_PYTHON_BIN:-}" ]]; then module load python-cbrg; fi
    exec env PYTHONNOUSERSITE=1 MPLBACKEND=Agg "${REPORT_PYTHON_BIN:-python3}" "${WORKSPACE}/src/pipeline.py" "$@"
    ;;
  rebuild-inputs)
    if command -v module >/dev/null 2>&1; then module load R-cbrg; module load cuda/11.8; fi
    ;;
  *)
    if command -v module >/dev/null 2>&1; then module load cuda/11.8; fi
    ;;
esac
exec env -u PYTHONPATH PYTHONNOUSERSITE=1 "${PYTHON_BIN:-python3}" \
  "${WORKSPACE}/src/pipeline.py" "$@"
