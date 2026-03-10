#!/bin/bash
# Auto-launch Round 3 after Round 2 test group completes
# Waits for v2_test run to finish, then launches R3 control + test sequentially

cd /home/jason/clawd/karpathy-problem
source .venv/bin/activate

echo "[$(date)] Waiting for Round 2 test group to complete..."

# Wait for the orchestrator process to finish
while pgrep -f "orchestrator.py --mode test" > /dev/null 2>&1; do
    sleep 60
done

echo "[$(date)] Round 2 test group finished. Starting Round 3 in 60 seconds (GPU cooldown)..."
sleep 60

# Round 3 Control: LLM + TSV
echo "[$(date)] Launching Round 3 CONTROL (LLM + TSV)..."
PYTHONUNBUFFERED=1 python3 orchestrator.py --mode control --hours 8 --reasoning llm \
    > results/v3_control_run.log 2>&1

echo "[$(date)] Round 3 Control finished. Starting test group in 60 seconds..."
sleep 60

# Round 3 Test: LLM + Engram
echo "[$(date)] Launching Round 3 TEST (LLM + Engram)..."
PYTHONUNBUFFERED=1 python3 orchestrator.py --mode test --hours 8 --reasoning llm \
    > results/v3_test_run.log 2>&1

echo "[$(date)] Round 3 complete! Both groups finished."
