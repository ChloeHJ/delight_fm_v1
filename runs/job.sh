#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="${V12_RUN_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
exec bash "${RUN_DIR}/run.sh" "$@"
