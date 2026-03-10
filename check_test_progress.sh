#!/bin/bash
LOG="/home/jason/clawd/karpathy-problem/results/v3_test_run.log"
STATE="/home/jason/clawd/karpathy-problem/results/.last_check_round"

# Get current round
CURRENT_ROUND=$(grep "^ROUND" "$LOG" 2>/dev/null | tail -1 | awk '{print $2}')
LAST_ROUND=$(cat "$STATE" 2>/dev/null || echo "0")

# Get latest improvements
IMPROVEMENTS=$(grep "IMPROVEMENT" "$LOG" 2>/dev/null | wc -l)
BEST=$(grep "IMPROVEMENT" "$LOG" 2>/dev/null | tail -1 | grep -oP 'val_bpb=\K[0-9.]+')

echo "$CURRENT_ROUND" > "$STATE"
echo "round=$CURRENT_ROUND improvements=$IMPROVEMENTS best=$BEST"
