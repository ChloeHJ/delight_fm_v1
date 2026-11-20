#!/usr/bin/env bash
set -euo pipefail
GIT_ROOT="${DELIGHT_FM_GIT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
exec bash "${GIT_ROOT}/runs/run.sh" "$@"
