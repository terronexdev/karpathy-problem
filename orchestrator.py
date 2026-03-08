#!/usr/bin/env python3
"""
Autoresearch Orchestrator — Runs the Karpathy Problem experiment.

Two modes:
  CONTROL: Agents pick hyperparams using results.tsv (grep/heuristic)
  TEST:    Agents pick hyperparams using Engram semantic memory

Each iteration:
  1. Propose a hyperparameter change (informed by prior results)
  2. Patch train.py with the new config
  3. Run training (5-min budget)
  4. Evaluate val_bpb
  5. Keep or discard the change
  6. Record the result
  7. Repeat

Usage:
  python orchestrator.py --mode control --hours 1
  python orchestrator.py --mode test --hours 1
"""

import os
import sys
import json
import time
import random
import shutil
import subprocess
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional

SCRIPT_DIR = Path(__file__).parent
TRAIN_SCRIPT = SCRIPT_DIR / "train.py"
TRAIN_BACKUP = SCRIPT_DIR / "train.py.baseline"
RESULTS_DIR = SCRIPT_DIR / "results"

# Tunable hyperparameters and their ranges
# Only tune safe hyperparameters — LR and optimization params.
# DEPTH and HEAD_DIM change tensor shapes and require architecture-aware handling.
# DEVICE_BATCH_SIZE and TOTAL_BATCH_SIZE affect memory, keep fixed for fair comparison.
HYPERPARAMS = {
    "EMBEDDING_LR":   {"min": 0.1, "max": 2.0, "type": float, "default": 0.6},
    "UNEMBEDDING_LR": {"min": 0.001, "max": 0.02, "type": float, "default": 0.004},
    "MATRIX_LR":      {"min": 0.005, "max": 0.2, "type": float, "default": 0.04},
    "SCALAR_LR":      {"min": 0.1, "max": 2.0, "type": float, "default": 0.5},
    "WEIGHT_DECAY":   {"min": 0.0, "max": 0.5, "type": float, "default": 0.2},
    "WARMUP_RATIO":   {"min": 0.0, "max": 0.15, "type": float, "default": 0.0},
    "WARMDOWN_RATIO":  {"min": 0.2, "max": 0.8, "type": float, "default": 0.5},
}


@dataclass
class ExperimentResult:
    iteration: int
    param_name: str
    old_value: str
    new_value: str
    val_bpb: float
    status: str  # keep, discard, crash
    description: str
    reasoning: str
    duration_s: float
    vram_mb: float = 0.0
    recall_context: str = ""


class Orchestrator:
    def __init__(self, mode: str, hours: float, agent_id: str = "agent-1"):
        self.mode = mode  # "control" or "test"
        self.hours = hours
        self.agent_id = agent_id
        self.best_bpb = 1.189370  # baseline
        self.current_config = {k: v["default"] for k, v in HYPERPARAMS.items()}
        self.history: list[ExperimentResult] = []
        self.start_time = None

        # Set up results directory
        self.run_id = f"{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir = RESULTS_DIR / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Backup the original train.py
        if not TRAIN_BACKUP.exists():
            shutil.copy2(TRAIN_SCRIPT, TRAIN_BACKUP)

    def run(self):
        """Main experiment loop."""
        self.start_time = time.time()
        end_time = self.start_time + (self.hours * 3600)
        iteration = 0

        print(f"\n{'='*60}")
        print(f"KARPATHY PROBLEM EXPERIMENT")
        print(f"Mode: {self.mode.upper()}")
        print(f"Duration: {self.hours} hours")
        print(f"Baseline val_bpb: {self.best_bpb:.6f}")
        print(f"Run ID: {self.run_id}")
        print(f"{'='*60}\n")

        try:
            while time.time() < end_time:
                iteration += 1
                remaining = (end_time - time.time()) / 60
                print(f"\n--- Iteration {iteration} | {remaining:.0f} min remaining ---")

                # Check if we have enough time for another run (~6 min)
                if remaining < 7:
                    print("Not enough time for another iteration. Stopping.")
                    break

                # 1. Decide what to try
                param, old_val, new_val, reasoning, recall_ctx = self.propose_change(iteration)

                # 2. Apply the change — always start from clean baseline
                shutil.copy2(TRAIN_BACKUP, TRAIN_SCRIPT)
                # Apply all current best config values
                for p, v in self.current_config.items():
                    if p in HYPERPARAMS:
                        self.patch_train_script(p, v)
                # Then apply the experimental change
                self.patch_train_script(param, new_val)

                # 3. Train
                print(f"Training with {param}={new_val} (was {old_val})...")
                t0 = time.time()
                val_bpb, vram_mb, crashed = self.run_training()
                duration = time.time() - t0

                # 4. Evaluate
                if crashed:
                    status = "crash"
                    print(f"  CRASHED after {duration:.0f}s")
                elif val_bpb < self.best_bpb:
                    status = "keep"
                    improvement = self.best_bpb - val_bpb
                    print(f"  IMPROVEMENT! val_bpb={val_bpb:.6f} (delta={improvement:.6f})")
                    self.best_bpb = val_bpb
                    self.current_config[param] = new_val
                else:
                    status = "discard"
                    print(f"  No improvement. val_bpb={val_bpb:.6f} vs best={self.best_bpb:.6f}")

                # 5. Record
                result = ExperimentResult(
                    iteration=iteration,
                    param_name=param,
                    old_value=str(old_val),
                    new_value=str(new_val),
                    val_bpb=val_bpb if not crashed else 999.0,
                    status=status,
                    description=f"Changed {param} from {old_val} to {new_val}",
                    reasoning=reasoning,
                    duration_s=duration,
                    vram_mb=vram_mb,
                    recall_context=recall_ctx,
                )
                self.history.append(result)
                self.record_result(result)

        except KeyboardInterrupt:
            print("\nExperiment interrupted by user.")
        finally:
            # Restore original train.py
            if TRAIN_BACKUP.exists():
                shutil.copy2(TRAIN_BACKUP, TRAIN_SCRIPT)

            self.save_summary()

    def propose_change(self, iteration: int):
        """Propose a hyperparameter change. Strategy differs by mode."""
        if self.mode == "test":
            return self._propose_with_engram(iteration)
        else:
            return self._propose_with_tsv(iteration)

    def _propose_with_tsv(self, iteration: int):
        """Control group: pick changes using results.tsv and simple heuristics."""
        # Simple strategy: cycle through params, try random perturbations
        params = list(HYPERPARAMS.keys())
        param = params[iteration % len(params)]
        spec = HYPERPARAMS[param]
        old_val = self.current_config[param]

        # Random perturbation
        if "values" in spec:
            candidates = [v for v in spec["values"] if v != old_val]
            new_val = random.choice(candidates) if candidates else old_val
        elif spec["type"] == int:
            delta = max(1, int(old_val * 0.3))
            new_val = old_val + random.choice([-delta, delta])
            new_val = max(spec["min"], min(spec["max"], new_val))
        else:
            factor = random.choice([0.5, 0.7, 1.5, 2.0])
            new_val = round(old_val * factor, 6)
            new_val = max(spec["min"], min(spec["max"], new_val))

        # Simple reasoning from TSV
        reasoning = f"Iteration {iteration}: trying {param} change. "
        kept = [r for r in self.history if r.status == "keep"]
        if kept:
            reasoning += f"Best so far: {self.best_bpb:.6f} from iteration {kept[-1].iteration}."
        else:
            reasoning += "No improvements yet, exploring."

        return param, old_val, new_val, reasoning, ""

    def _propose_with_engram(self, iteration: int):
        """Test group: use Engram semantic memory to inform decisions."""
        params = list(HYPERPARAMS.keys())
        param = params[iteration % len(params)]
        spec = HYPERPARAMS[param]
        old_val = self.current_config[param]

        # Query Engram for relevant prior knowledge
        query = f"experiments changing {param} learning rate optimization"
        recall_ctx = ""
        try:
            result = subprocess.run(
                ["python3", str(SCRIPT_DIR / "research_memory.py"), "recall", query, "--limit", "3"],
                capture_output=True, text=True, timeout=15, cwd=str(SCRIPT_DIR)
            )
            recall_ctx = result.stdout.strip()
        except Exception as e:
            recall_ctx = f"Recall failed: {e}"

        # Parse recall context for intelligent decisions
        reasoning = f"Iteration {iteration}: Engram recall for '{param}'.\n"

        # Analyze recall results to pick direction
        direction = self._analyze_recall_for_direction(param, recall_ctx)
        reasoning += f"Engram insight: {direction['reasoning']}\n"

        if "values" in spec:
            if direction["go_higher"]:
                candidates = [v for v in spec["values"] if v > old_val]
            else:
                candidates = [v for v in spec["values"] if v < old_val]
            if not candidates:
                candidates = [v for v in spec["values"] if v != old_val]
            new_val = random.choice(candidates) if candidates else old_val
        elif spec["type"] == int:
            delta = max(1, int(old_val * 0.25))
            new_val = old_val + delta if direction["go_higher"] else old_val - delta
            new_val = max(spec["min"], min(spec["max"], new_val))
        else:
            factor = direction["factor"]
            new_val = round(old_val * factor, 6)
            new_val = max(spec["min"], min(spec["max"], new_val))

        return param, old_val, new_val, reasoning, recall_ctx

    def _analyze_recall_for_direction(self, param: str, recall_ctx: str):
        """Analyze Engram recall results to decide which direction to go."""
        # Look for patterns in recall context
        lower = recall_ctx.lower()

        # Check if increasing this param led to improvements
        increase_good = ("increas" in lower and "improv" in lower) or \
                       ("higher" in lower and "better" in lower)
        decrease_good = ("decreas" in lower and "improv" in lower) or \
                       ("lower" in lower and "better" in lower) or \
                       ("reduc" in lower and "improv" in lower)

        if increase_good and not decrease_good:
            return {"go_higher": True, "factor": 1.5, "reasoning": "Prior experiments suggest increasing helps"}
        elif decrease_good and not increase_good:
            return {"go_higher": False, "factor": 0.7, "reasoning": "Prior experiments suggest decreasing helps"}
        elif "crash" in lower or "oom" in lower:
            return {"go_higher": False, "factor": 0.8, "reasoning": "Prior crashes suggest being conservative"}
        else:
            # No clear signal — explore both directions but with smaller steps
            go_higher = random.random() > 0.5
            return {
                "go_higher": go_higher,
                "factor": 1.3 if go_higher else 0.75,
                "reasoning": "No clear prior signal, exploring with moderate step"
            }

    def patch_train_script(self, param: str, new_value):
        """Patch train.py with a new hyperparameter value."""
        content = TRAIN_SCRIPT.read_text()
        spec = HYPERPARAMS[param]

        # Find and replace the parameter line
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.startswith(f"{param}") and "=" in stripped:
                # Format the new value
                if spec["type"] == int:
                    if param == "TOTAL_BATCH_SIZE":
                        # Keep power-of-2 notation
                        import math
                        log2 = math.log2(new_value)
                        if log2 == int(log2):
                            val_str = f"2**{int(log2)}"
                        else:
                            val_str = str(new_value)
                    else:
                        val_str = str(new_value)
                else:
                    val_str = str(new_value)

                # Preserve the comment if any
                parts = line.split("#", 1)
                indent = line[:len(line) - len(line.lstrip())]
                if len(parts) > 1:
                    new_line = f"{indent}{param} = {val_str:<16s} # {parts[1].strip()}"
                else:
                    new_line = f"{indent}{param} = {val_str}"

                content = content.replace(line, new_line, 1)
                break

        TRAIN_SCRIPT.write_text(content)

    def run_training(self):
        """Execute train.py and capture results."""
        log_file = self.run_dir / f"iter_{len(self.history)+1}.log"

        try:
            result = subprocess.run(
                [sys.executable, str(TRAIN_SCRIPT)],
                capture_output=True, text=True,
                timeout=600,  # 10 min hard limit
                cwd=str(SCRIPT_DIR),
                env={**os.environ, "PYTORCH_ALLOC_CONF": "expandable_segments:True"}
            )

            output = result.stdout + result.stderr
            log_file.write_text(output)

            if result.returncode != 0:
                return 999.0, 0.0, True

            # Parse val_bpb from output
            val_bpb = self._parse_val_bpb(output)
            vram_mb = self._parse_vram(output)

            return val_bpb, vram_mb, False

        except subprocess.TimeoutExpired:
            log_file.write_text("TIMEOUT after 600s")
            return 999.0, 0.0, True
        except Exception as e:
            log_file.write_text(f"ERROR: {e}")
            return 999.0, 0.0, True

    def _parse_val_bpb(self, output: str) -> float:
        """Extract val_bpb from training output."""
        for line in output.split("\n"):
            if "val_bpb:" in line:
                try:
                    return float(line.split("val_bpb:")[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
        return 999.0

    def _parse_vram(self, output: str) -> float:
        """Extract peak VRAM from training output."""
        for line in output.split("\n"):
            if "peak_vram_mb:" in line:
                try:
                    return float(line.split("peak_vram_mb:")[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
        return 0.0

    def record_result(self, result: ExperimentResult):
        """Record to Engram (test mode) or just TSV (control mode)."""
        # Always record to local JSON log
        log_path = self.run_dir / "experiments.jsonl"
        with open(log_path, "a") as f:
            f.write(json.dumps({
                "iteration": result.iteration,
                "param": result.param_name,
                "old": result.old_value,
                "new": result.new_value,
                "val_bpb": result.val_bpb,
                "status": result.status,
                "description": result.description,
                "reasoning": result.reasoning,
                "duration_s": result.duration_s,
                "vram_mb": result.vram_mb,
                "mode": self.mode,
                "timestamp": datetime.now().isoformat(),
            }) + "\n")

        if self.mode == "test":
            # Also record to Engram
            try:
                subprocess.run(
                    ["python3", str(SCRIPT_DIR / "research_memory.py"), "remember",
                     "--commit", f"iter-{result.iteration}",
                     "--val-bpb", str(result.val_bpb),
                     "--vram-gb", str(result.vram_mb / 1024),
                     "--status", result.status,
                     "--description", result.description,
                     "--agent", self.agent_id,
                     "--reasoning", result.reasoning],
                    capture_output=True, text=True, timeout=15,
                    cwd=str(SCRIPT_DIR)
                )
            except Exception as e:
                print(f"  Warning: Engram record failed: {e}")

    def save_summary(self):
        """Save final experiment summary."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        total = len(self.history)
        keeps = sum(1 for r in self.history if r.status == "keep")
        crashes = sum(1 for r in self.history if r.status == "crash")

        summary = {
            "run_id": self.run_id,
            "mode": self.mode,
            "hours_planned": self.hours,
            "hours_actual": elapsed / 3600,
            "total_iterations": total,
            "improvements": keeps,
            "crashes": crashes,
            "hit_rate": keeps / total if total > 0 else 0,
            "baseline_bpb": 1.189370,
            "best_bpb": self.best_bpb,
            "improvement_delta": 1.189370 - self.best_bpb,
            "final_config": self.current_config,
        }

        summary_path = self.run_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))

        print(f"\n{'='*60}")
        print(f"EXPERIMENT COMPLETE: {self.run_id}")
        print(f"{'='*60}")
        print(f"Mode:           {self.mode.upper()}")
        print(f"Duration:       {elapsed/60:.1f} minutes")
        print(f"Iterations:     {total}")
        print(f"Improvements:   {keeps} ({keeps/total*100:.0f}% hit rate)" if total else "")
        print(f"Crashes:        {crashes}")
        print(f"Baseline BPB:   1.189370")
        print(f"Best BPB:       {self.best_bpb:.6f}")
        print(f"Improvement:    {1.189370 - self.best_bpb:.6f}")
        print(f"Results:        {self.run_dir}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Karpathy Problem Orchestrator")
    parser.add_argument("--mode", choices=["control", "test"], required=True,
                       help="control=TSV only, test=Engram semantic memory")
    parser.add_argument("--hours", type=float, default=1.0,
                       help="Duration in hours (default: 1)")
    parser.add_argument("--agent", default="agent-1",
                       help="Agent identifier")
    args = parser.parse_args()

    orchestrator = Orchestrator(args.mode, args.hours, args.agent)
    orchestrator.run()


if __name__ == "__main__":
    main()
