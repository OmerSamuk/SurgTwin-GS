#!/usr/bin/env bash
# run_m4a2_0.sh — forwarded to canonical 20K smoke script
# This file delegates to run_m4a2_0_20k.sh (the canonical M4-A2-0 smoke).
# Retained as a forwarder for backwards compatibility.
#
# For the original 50K diagnostic run, see run_m4a1r_50k_warmup.sh.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_m4a2_0_20k.sh" "$@"
