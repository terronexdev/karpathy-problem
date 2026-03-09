#!/usr/bin/env python3
"""
Autoresearch Orchestrator v2 — Multi-Agent Karpathy Problem Experiment.

The Karpathy Problem: Git assumes convergence (branches merge to master).
Multi-agent research needs divergent exploration with knowledge sharing.

This orchestrator tests whether Engram semantic memory enables better
multi-agent collaboration than git's branch/merge model (simulated via TSVs).

Two modes:
  CONTROL: 3 agents, each with own TSV. After each round, TSVs are merged
           (concatenated) simulating git merge. Agents grep merged file.
  TEST:    3 agents, one shared Engram. Each agent writes results and queries
           semantically before each experiment — no merge step needed.

Agents are interleaved sequentially (Agent1 → Agent2 → Agent3 → repeat)
to simulate parallelism on a single GPU.

Usage:
  python orchestrator.py --mode control --hours 8
  python orchestrator.py --mode test --hours 8
  python orchestrator.py --mode control --hours 1 --agents 3
"""

import os
import sys
import json
import time
import random
import shutil
import subprocess
import argparse
import csv
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

SCRIPT_DIR = Path(__file__).parent
TRAIN_SCRIPT = SCRIPT_DIR / "train.py"
TRAIN_BACKUP = SCRIPT_DIR / "train.py.baseline"
RESULTS_DIR = SCRIPT_DIR / "results"

# Tunable hyperparameters — safe LR and optimization params only
HYPERPARAMS = {
    "EMBEDDING_LR":   {"min": 0.1, "max": 2.0, "type": float, "default": 0.6},
    "UNEMBEDDING_LR": {"min": 0.001, "max": 0.02, "type": float, "default": 0.004},
    "MATRIX_LR":      {"min": 0.005, "max": 0.2, "type": float, "default": 0.04},
    "SCALAR_LR":      {"min": 0.1, "max": 2.0, "type": float, "default": 0.5},
    "WEIGHT_DECAY":   {"min": 0.0, "max": 0.5, "type": float, "default": 0.2},
    "WARMUP_RATIO":   {"min": 0.0, "max": 0.15, "type": float, "default": 0.0},
    "WARMDOWN_RATIO":  {"min": 0.2, "max": 0.8, "type": float, "default": 0.5},
}

# Each agent has a "personality" — a preferred exploration strategy
# This creates divergent exploration, which is the whole point
AGENT_STRATEGIES = {
    "agent-1": {
        "name": "Explorer",
        "desc": "Aggressive — tries large perturbations, favors unexplored regions",
        "factor_range": (0.4, 0.6, 1.8, 2.5),  # (low_min, low_max, high_min, high_max)
        "param_preference": None,  # explores all params equally
    },
    "agent-2": {
        "name": "Refiner",
        "desc": "Conservative — small perturbations around best known config",
        "factor_range": (0.8, 0.95, 1.05, 1.25),
        "param_preference": None,
    },
    "agent-3": {
        "name": "Specialist",
        "desc": "Focuses on learning rates, tries combinations",
        "factor_range": (0.6, 0.8, 1.3, 1.6),
        "param_preference": ["EMBEDDING_LR", "UNEMBEDDING_LR", "MATRIX_LR", "SCALAR_LR"],
    },
}


@dataclass
class ExperimentResult:
    iteration: int
    round_num: int
    agent_id: str
    param_name: str
    old_value: str
    new_value: str
    val_bpb: float
    status: str
    description: str
    reasoning: str
    duration_s: float
    vram_mb: float = 0.0
    recall_context: str = ""


class Agent:
    """Represents one research agent with its own state and exploration strategy."""

    def __init__(self, agent_id: str, strategy: dict):
        self.agent_id = agent_id
        self.strategy = strategy
        self.best_bpb = 1.189370  # each agent tracks its own best
        self.current_config = {k: v["default"] for k, v in HYPERPARAMS.items()}
        self.history: list[ExperimentResult] = []
        self.iteration = 0
        self.improvements = 0

    def pick_param(self):
        """Pick which parameter to tune, based on agent personality."""
        prefs = self.strategy.get("param_preference")
        if prefs:
            # Specialist: 70% chance to pick preferred params
            if random.random() < 0.7:
                return random.choice(prefs)
        params = list(HYPERPARAMS.keys())
        return random.choice(params)

    def pick_value(self, param: str, direction_hint: dict = None):
        """Pick a new value, based on agent personality and optional hint."""
        spec = HYPERPARAMS[param]
        old_val = self.current_config[param]
        fmin_lo, fmax_lo, fmin_hi, fmax_hi = self.strategy["factor_range"]

        if direction_hint and direction_hint.get("go_higher") is not None:
            if direction_hint["go_higher"]:
                factor = random.uniform(fmin_hi, fmax_hi)
            else:
                factor = random.uniform(fmin_lo, fmax_lo)
        else:
            # Random direction
            if random.random() > 0.5:
                factor = random.uniform(fmin_hi, fmax_hi)
            else:
                factor = random.uniform(fmin_lo, fmax_lo)

        new_val = round(old_val * factor, 6)
        new_val = max(spec["min"], min(spec["max"], new_val))
        return new_val


class MultiAgentOrchestrator:
    """Orchestrates multiple agents in interleaved rounds."""

    def __init__(self, mode: str, hours: float, num_agents: int = 3):
        self.mode = mode
        self.hours = hours
        self.num_agents = num_agents
        self.start_time = None

        # Create agents
        agent_ids = list(AGENT_STRATEGIES.keys())[:num_agents]
        self.agents = {
            aid: Agent(aid, AGENT_STRATEGIES[aid])
            for aid in agent_ids
        }

        # Set up results directory
        self.run_id = f"v2_{mode}_{num_agents}agents_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir = RESULTS_DIR / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Control mode: each agent has its own TSV
        if mode == "control":
            self.tsv_dir = self.run_dir / "tsvs"
            self.tsv_dir.mkdir(exist_ok=True)
            for aid in self.agents:
                tsv_path = self.tsv_dir / f"{aid}.tsv"
                tsv_path.write_text("agent\tparam\told\tnew\tval_bpb\tstatus\tdescription\n")
            # Merged TSV (simulates git merge)
            (self.tsv_dir / "merged.tsv").write_text(
                "agent\tparam\told\tnew\tval_bpb\tstatus\tdescription\n"
            )

        # Test mode: clear Engram for fresh start
        if mode == "test":
            engram_path = Path.home() / ".karpathy-problem" / "research-brain.engram"
            if engram_path.exists():
                engram_path.unlink()

        # Backup train.py
        if not TRAIN_BACKUP.exists():
            shutil.copy2(TRAIN_SCRIPT, TRAIN_BACKUP)

        # Global tracking
        self.all_results: list[ExperimentResult] = []
        self.total_iterations = 0

    def run(self):
        """Main experiment loop — interleave agents in rounds."""
        self.start_time = time.time()
        end_time = self.start_time + (self.hours * 3600)
        round_num = 0

        agent_ids = list(self.agents.keys())

        print(f"\n{'='*60}")
        print(f"KARPATHY PROBLEM v2 — MULTI-AGENT EXPERIMENT")
        print(f"{'='*60}")
        print(f"Mode:     {self.mode.upper()}")
        print(f"Agents:   {self.num_agents}")
        for aid, agent in self.agents.items():
            print(f"  {aid}: {agent.strategy['name']} — {agent.strategy['desc']}")
        print(f"Duration: {self.hours} hours")
        print(f"Baseline: 1.189370")
        print(f"Run ID:   {self.run_id}")
        print(f"{'='*60}\n")

        try:
            while time.time() < end_time:
                round_num += 1
                remaining_min = (end_time - time.time()) / 60

                print(f"\n{'='*40}")
                print(f"ROUND {round_num} | {remaining_min:.0f} min remaining")
                print(f"{'='*40}")

                for agent_id in agent_ids:
                    remaining_min = (end_time - time.time()) / 60
                    if remaining_min < 7:
                        print(f"Not enough time for {agent_id}. Stopping.")
                        raise StopIteration()

                    agent = self.agents[agent_id]
                    agent.iteration += 1
                    self.total_iterations += 1

                    print(f"\n--- {agent_id} ({agent.strategy['name']}) | "
                          f"iter {agent.iteration} | best={agent.best_bpb:.6f} ---")

                    # 1. Propose change
                    param, old_val, new_val, reasoning, recall_ctx = \
                        self.propose_change(agent, round_num)

                    # 2. Apply config from clean baseline
                    shutil.copy2(TRAIN_BACKUP, TRAIN_SCRIPT)
                    for p, v in agent.current_config.items():
                        if p in HYPERPARAMS:
                            self.patch_train_script(p, v)
                    self.patch_train_script(param, new_val)

                    # 3. Train
                    print(f"  Training: {param}={new_val} (was {old_val})")
                    t0 = time.time()
                    val_bpb, vram_mb, crashed = self.run_training()
                    duration = time.time() - t0

                    # 4. Evaluate
                    if crashed:
                        status = "crash"
                        print(f"  CRASHED after {duration:.0f}s")
                    elif val_bpb < agent.best_bpb:
                        status = "keep"
                        delta = agent.best_bpb - val_bpb
                        agent.best_bpb = val_bpb
                        agent.current_config[param] = new_val
                        agent.improvements += 1
                        print(f"  IMPROVEMENT! val_bpb={val_bpb:.6f} (delta={delta:.6f})")
                    else:
                        status = "discard"
                        print(f"  No improvement. {val_bpb:.6f} vs best={agent.best_bpb:.6f}")

                    # 5. Record
                    result = ExperimentResult(
                        iteration=agent.iteration,
                        round_num=round_num,
                        agent_id=agent_id,
                        param_name=param,
                        old_value=str(old_val),
                        new_value=str(new_val),
                        val_bpb=val_bpb if not crashed else 999.0,
                        status=status,
                        description=f"{agent_id} changed {param}: {old_val} -> {new_val}",
                        reasoning=reasoning,
                        duration_s=duration,
                        vram_mb=vram_mb,
                        recall_context=recall_ctx,
                    )
                    agent.history.append(result)
                    self.all_results.append(result)
                    self.record_result(result, agent)

                # After each round in control mode: merge TSVs (simulate git merge)
                if self.mode == "control":
                    self.merge_tsvs()
                    print(f"\n  [Git merge] Combined {self.num_agents} agent TSVs -> merged.tsv")

        except (KeyboardInterrupt, StopIteration):
            pass
        finally:
            if TRAIN_BACKUP.exists():
                shutil.copy2(TRAIN_BACKUP, TRAIN_SCRIPT)
            self.save_summary()

    def propose_change(self, agent: Agent, round_num: int):
        """Propose a hyperparameter change based on mode and agent strategy."""
        param = agent.pick_param()
        old_val = agent.current_config[param]

        if self.mode == "test":
            return self._propose_with_engram(agent, param, old_val, round_num)
        else:
            return self._propose_with_tsv(agent, param, old_val, round_num)

    def _propose_with_tsv(self, agent: Agent, param: str, old_val, round_num: int):
        """Control: read merged TSV, grep for relevant experiments."""
        recall_ctx = ""
        reasoning = f"Round {round_num}, {agent.agent_id} ({agent.strategy['name']}): "

        # Read merged TSV and look for this param
        merged_path = self.tsv_dir / "merged.tsv"
        relevant_rows = []
        try:
            with open(merged_path) as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    if row.get("param") == param:
                        relevant_rows.append(row)
        except Exception:
            pass

        direction_hint = None
        if relevant_rows:
            # Simple heuristic: look at what other agents found
            keeps = [r for r in relevant_rows if r.get("status") == "keep"]
            discards = [r for r in relevant_rows if r.get("status") == "discard"]

            reasoning += f"Found {len(relevant_rows)} prior experiments with {param} "
            reasoning += f"({len(keeps)} kept, {len(discards)} discarded). "

            if keeps:
                # Try to move in the direction of successful experiments
                last_keep = keeps[-1]
                try:
                    kept_old = float(last_keep["old"])
                    kept_new = float(last_keep["new"])
                    went_higher = kept_new > kept_old
                    direction_hint = {
                        "go_higher": went_higher,
                        "reasoning": f"Agent {last_keep.get('agent', '?')} improved by "
                                     f"{'increasing' if went_higher else 'decreasing'} {param}"
                    }
                    reasoning += f"Following {last_keep.get('agent', '?')}'s successful direction. "
                except (ValueError, KeyError):
                    pass

            if not direction_hint and discards:
                # Avoid directions that failed
                last_discard = discards[-1]
                try:
                    disc_old = float(last_discard["old"])
                    disc_new = float(last_discard["new"])
                    went_higher = disc_new > disc_old
                    direction_hint = {
                        "go_higher": not went_higher,  # opposite of what failed
                        "reasoning": f"Avoiding direction that failed for {last_discard.get('agent', '?')}"
                    }
                    reasoning += f"Avoiding failed direction. "
                except (ValueError, KeyError):
                    pass

            recall_ctx = json.dumps(relevant_rows[-3:], indent=2)  # last 3 relevant
        else:
            reasoning += f"No prior experiments with {param} in merged TSV. Exploring. "

        new_val = agent.pick_value(param, direction_hint)
        return param, old_val, new_val, reasoning, recall_ctx

    def _propose_with_engram(self, agent: Agent, param: str, old_val, round_num: int):
        """Test: query shared Engram for semantically relevant experiments."""
        # Build a rich semantic query
        queries = [
            f"experiments changing {param} results improvements failures",
            f"what happened when agents modified {param} learning rate",
            f"best hyperparameter configurations for {param}",
        ]
        query = random.choice(queries)

        recall_ctx = ""
        try:
            result = subprocess.run(
                ["python3", str(SCRIPT_DIR / "research_memory.py"), "recall", query,
                 "--limit", "5"],
                capture_output=True, text=True, timeout=15, cwd=str(SCRIPT_DIR)
            )
            recall_ctx = result.stdout.strip()
        except Exception as e:
            recall_ctx = f"Recall failed: {e}"

        reasoning = f"Round {round_num}, {agent.agent_id} ({agent.strategy['name']}): "
        reasoning += f"Engram query: '{query}'\n"

        # Analyze recall for direction
        direction_hint = self._analyze_recall(param, recall_ctx)
        reasoning += f"Insight: {direction_hint.get('reasoning', 'no signal')}\n"

        # Count how many results came back
        result_count = recall_ctx.count("[") if recall_ctx else 0
        if result_count > 0:
            reasoning += f"Found {result_count} semantically relevant prior experiments. "
        else:
            reasoning += "No relevant prior experiments in Engram yet. "

        new_val = agent.pick_value(param, direction_hint)
        return param, old_val, new_val, reasoning, recall_ctx

    def _analyze_recall(self, param: str, recall_ctx: str):
        """Analyze Engram recall results to inform direction."""
        if not recall_ctx or len(recall_ctx) < 10:
            return {"go_higher": None, "reasoning": "No data to analyze"}

        lower = recall_ctx.lower()

        # Look for explicit improvement signals
        increase_good = ("increas" in lower and ("improv" in lower or "keep" in lower)) or \
                       ("higher" in lower and "better" in lower)
        decrease_good = ("decreas" in lower and ("improv" in lower or "keep" in lower)) or \
                       ("lower" in lower and "better" in lower) or \
                       ("reduc" in lower and ("improv" in lower or "keep" in lower))
        crash_signal = "crash" in lower or "oom" in lower

        # Look for specific val_bpb improvements
        if "keep" in lower:
            # Try to figure out which direction the "keep" went
            if f"increased {param.lower()}" in lower or f"{param.lower()} from" in lower:
                increase_good = True

        if crash_signal:
            return {"go_higher": False, "reasoning": "Prior crashes — being conservative"}
        elif increase_good and not decrease_good:
            return {"go_higher": True, "reasoning": f"Prior experiments show increasing {param} helps"}
        elif decrease_good and not increase_good:
            return {"go_higher": False, "reasoning": f"Prior experiments show decreasing {param} helps"}
        elif increase_good and decrease_good:
            return {"go_higher": None, "reasoning": "Mixed signals — exploring freely"}
        else:
            return {"go_higher": None, "reasoning": "No clear directional signal"}

    def merge_tsvs(self):
        """Simulate git merge: concatenate all agent TSVs into merged.tsv."""
        merged_path = self.tsv_dir / "merged.tsv"
        header = "agent\tparam\told\tnew\tval_bpb\tstatus\tdescription\n"
        rows = []

        for aid in self.agents:
            tsv_path = self.tsv_dir / f"{aid}.tsv"
            try:
                with open(tsv_path) as f:
                    for i, line in enumerate(f):
                        if i == 0:  # skip header
                            continue
                        rows.append(line)
            except Exception:
                pass

        with open(merged_path, 'w') as f:
            f.write(header)
            for row in rows:
                f.write(row)

    def record_result(self, result: ExperimentResult, agent: Agent):
        """Record result to appropriate storage."""
        # Always write to JSONL
        log_path = self.run_dir / "experiments.jsonl"
        with open(log_path, "a") as f:
            f.write(json.dumps({
                "round": result.round_num,
                "iteration": result.iteration,
                "agent": result.agent_id,
                "agent_strategy": agent.strategy["name"],
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

        if self.mode == "control":
            # Write to agent's own TSV
            tsv_path = self.tsv_dir / f"{result.agent_id}.tsv"
            with open(tsv_path, "a") as f:
                f.write(f"{result.agent_id}\t{result.param_name}\t{result.old_value}\t"
                        f"{result.new_value}\t{result.val_bpb:.6f}\t{result.status}\t"
                        f"{result.description}\n")

        elif self.mode == "test":
            # Write to shared Engram
            try:
                subprocess.run(
                    ["python3", str(SCRIPT_DIR / "research_memory.py"), "remember",
                     "--commit", f"r{result.round_num}-{result.agent_id}-i{result.iteration}",
                     "--val-bpb", str(result.val_bpb),
                     "--vram-gb", str(result.vram_mb / 1024),
                     "--status", result.status,
                     "--description", result.description,
                     "--agent", result.agent_id,
                     "--reasoning", result.reasoning],
                    capture_output=True, text=True, timeout=15,
                    cwd=str(SCRIPT_DIR)
                )
            except Exception as e:
                print(f"  Warning: Engram record failed: {e}")

    def patch_train_script(self, param: str, new_value):
        """Patch train.py with a new hyperparameter value."""
        content = TRAIN_SCRIPT.read_text()
        spec = HYPERPARAMS[param]

        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.startswith(f"{param}") and "=" in stripped:
                val_str = str(new_value)
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
        log_file = self.run_dir / f"iter_{self.total_iterations}.log"

        try:
            result = subprocess.run(
                [sys.executable, str(TRAIN_SCRIPT)],
                capture_output=True, text=True,
                timeout=600,
                cwd=str(SCRIPT_DIR),
                env={**os.environ, "PYTORCH_ALLOC_CONF": "expandable_segments:True"}
            )

            output = result.stdout + result.stderr
            log_file.write_text(output)

            if result.returncode != 0:
                return 999.0, 0.0, True

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
        for line in output.split("\n"):
            if "val_bpb:" in line:
                try:
                    return float(line.split("val_bpb:")[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
        return 999.0

    def _parse_vram(self, output: str) -> float:
        for line in output.split("\n"):
            if "peak_vram_mb:" in line:
                try:
                    return float(line.split("peak_vram_mb:")[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
        return 0.0

    def save_summary(self):
        """Save comprehensive multi-agent summary."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        total = len(self.all_results)
        total_keeps = sum(1 for r in self.all_results if r.status == "keep")
        total_crashes = sum(1 for r in self.all_results if r.status == "crash")

        # Per-agent stats
        agent_stats = {}
        for aid, agent in self.agents.items():
            agent_total = len(agent.history)
            agent_keeps = sum(1 for r in agent.history if r.status == "keep")
            agent_stats[aid] = {
                "strategy": agent.strategy["name"],
                "iterations": agent_total,
                "improvements": agent_keeps,
                "hit_rate": agent_keeps / agent_total if agent_total > 0 else 0,
                "best_bpb": agent.best_bpb,
                "improvement_delta": 1.189370 - agent.best_bpb,
                "final_config": agent.current_config,
            }

        # Global best
        global_best = min(a.best_bpb for a in self.agents.values())

        summary = {
            "run_id": self.run_id,
            "mode": self.mode,
            "num_agents": self.num_agents,
            "hours_planned": self.hours,
            "hours_actual": elapsed / 3600,
            "total_iterations": total,
            "total_improvements": total_keeps,
            "total_crashes": total_crashes,
            "overall_hit_rate": total_keeps / total if total > 0 else 0,
            "baseline_bpb": 1.189370,
            "global_best_bpb": global_best,
            "global_improvement": 1.189370 - global_best,
            "agents": agent_stats,
        }

        summary_path = self.run_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))

        # Print results
        print(f"\n{'='*60}")
        print(f"EXPERIMENT COMPLETE: {self.run_id}")
        print(f"{'='*60}")
        print(f"Mode:             {self.mode.upper()}")
        print(f"Agents:           {self.num_agents}")
        print(f"Duration:         {elapsed/60:.1f} minutes")
        print(f"Total Iterations: {total}")
        print(f"Improvements:     {total_keeps} ({total_keeps/total*100:.0f}% hit rate)" if total else "")
        print(f"Crashes:          {total_crashes}")
        print(f"Baseline BPB:     1.189370")
        print(f"Global Best BPB:  {global_best:.6f}")
        print(f"Improvement:      {1.189370 - global_best:.6f}")
        print(f"\nPer-Agent Results:")
        for aid, stats in agent_stats.items():
            print(f"  {aid} ({stats['strategy']}): "
                  f"best={stats['best_bpb']:.6f}, "
                  f"hits={stats['improvements']}/{stats['iterations']} "
                  f"({stats['hit_rate']*100:.0f}%)")
        print(f"\nResults: {self.run_dir}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Karpathy Problem v2 — Multi-Agent Orchestrator")
    parser.add_argument("--mode", choices=["control", "test"], required=True,
                       help="control=separate TSVs with merge, test=shared Engram")
    parser.add_argument("--hours", type=float, default=8.0,
                       help="Duration in hours (default: 8)")
    parser.add_argument("--agents", type=int, default=3,
                       help="Number of agents (default: 3)")
    args = parser.parse_args()

    orchestrator = MultiAgentOrchestrator(args.mode, args.hours, args.agents)
    orchestrator.run()


if __name__ == "__main__":
    main()
