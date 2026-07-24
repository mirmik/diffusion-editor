#!/bin/bash
# Compatibility entry point; all model workers now share one environment.
set -euo pipefail
exec "$(dirname "$0")/setup-workers.sh" "$@"
