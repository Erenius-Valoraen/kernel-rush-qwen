#!/bin/bash
# Usage: ./bench.sh [modal_bench.py args...]   e.g. ./bench.sh --shapes 1x512x32 --check 0
# Streams to the terminal AND saves logs/bench_<time>.log; prints a summary at the end.
cd "$(dirname "$0")"; mkdir -p logs
LOG="logs/bench_$(date +%H%M%S).log"
PYTHONIOENCODING=utf-8 python -m modal run modal_bench.py "$@" 2>&1 | grep -v "Loading checkpoint\|libnvrtc" | tee "$LOG"
echo; echo "===== SUMMARY ($LOG) ====="; grep -E "calibration|total .*tok/s|correctness|Error|error" "$LOG" | cut -c1-300
