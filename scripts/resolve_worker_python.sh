#!/bin/bash
# Shared CPython 3.11 discovery for isolated worker installers.

is_regular_cpython_311() {
    "$1" -c '
import platform
import sys
raise SystemExit(
    platform.python_implementation() != "CPython"
    or sys.version_info[:2] != (3, 11)
    or "t" in getattr(sys, "abiflags", "")
)
' >/dev/null 2>&1
}

resolve_worker_python() {
    local configured="${1:-}"
    local candidate

    if [ -n "$configured" ]; then
        if is_regular_cpython_311 "$configured"; then
            printf '%s\n' "$configured"
            return 0
        fi
        echo "ERROR: Configured worker Python is not regular CPython 3.11: $configured" >&2
        return 2
    fi

    candidate="$(command -v python3.11 2>/dev/null || true)"
    if [ -n "$candidate" ] && is_regular_cpython_311 "$candidate"; then
        printf '%s\n' "$candidate"
        return 0
    fi

    if command -v pyenv >/dev/null 2>&1; then
        candidate="$(
            PYENV_VERSION=3.11 pyenv which python3.11 2>/dev/null || true
        )"
        if [ -n "$candidate" ] && is_regular_cpython_311 "$candidate"; then
            printf '%s\n' "$candidate"
            return 0
        fi
    fi

    echo "ERROR: Regular CPython 3.11 was not found." >&2
    echo "Install Python 3.11 or set the worker's *_BOOTSTRAP_PYTHON override." >&2
    return 2
}
