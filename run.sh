#!/bin/bash
cd "$(dirname "$0")"
exec ./venv/bin/python3 -m diffusion_editor.main "$@"
