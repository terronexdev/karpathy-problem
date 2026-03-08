#!/usr/bin/env python3
"""
Research Memory — Engram integration for autoresearch

Replaces results.tsv as the knowledge discovery layer.
Git still handles code versioning. Engram handles what was tried,
what worked, and why.

Usage:
    # Record an experiment result
    python research_memory.py remember \
        --commit a1b2c3d \
        --val-bpb 0.993 \
        --vram-gb 10.2 \
        --status keep \
        --description "Increased LR to 0.04" \
        --agent agent-1

    # Recall similar experiments before starting a new one
    python research_memory.py recall "learning rate experiments"

    # Get full experiment history
    python research_memory.py history

    # Stats summary
    python research_memory.py stats
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

RESEARCH_DIR = Path.home() / ".karpathy-problem"
ENGRAM_FILE = RESEARCH_DIR / "research-brain.engram"
RESULTS_TSV = Path(__file__).parent / "results.tsv"

# We use Engram via the Node.js CLI since @terronex/engram-trace is a JS package.
# This Python wrapper calls the engram-trace CLI for remember/recall operations.

ENGRAM_CLI = Path(__file__).parent / "engram_cli.js"


def ensure_dirs():
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)


def run_engram(action: str, **kwargs) -> dict:
    """Call the Engram CLI helper."""
    cmd = ["node", str(ENGRAM_CLI), action, json.dumps(kwargs)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"Engram error: {result.stderr}", file=sys.stderr)
            return {"error": result.stderr}
        return json.loads(result.stdout) if result.stdout.strip() else {}
    except Exception as e:
        print(f"Engram call failed: {e}", file=sys.stderr)
        return {"error": str(e)}


def remember_experiment(commit: str, val_bpb: float, vram_gb: float,
                       status: str, description: str, agent: str = "agent-1",
                       reasoning: str = "", recall_context: str = ""):
    """Record an experiment result to both Engram and results.tsv."""
    ensure_dirs()

    # 1. Build the memory content (human-readable summary)
    outcome = "improved" if status == "keep" else ("failed" if status == "crash" else "no improvement")
    content = (
        f"Experiment by {agent}: {description}. "
        f"Result: val_bpb={val_bpb:.6f}, VRAM={vram_gb:.1f}GB. "
        f"Outcome: {outcome}."
    )
    if reasoning:
        content += f" Reasoning: {reasoning}"

    # 2. Store in Engram
    metadata = {
        "commit": commit,
        "val_bpb": val_bpb,
        "vram_gb": vram_gb,
        "status": status,
        "description": description,
        "agent": agent,
        "timestamp": datetime.now().isoformat(),
        "reasoning": reasoning,
        "recall_context": recall_context,
    }

    tags = ["experiment", f"status:{status}", f"agent:{agent}"]
    if status == "keep":
        tags.append("improvement")
    importance = 0.9 if status == "keep" else 0.4

    result = run_engram("remember", content=content, tags=tags,
                       metadata=metadata, importance=importance)

    # 3. Also append to results.tsv (backward compatible)
    tsv_line = f"{commit}\t{val_bpb:.6f}\t{vram_gb:.1f}\t{status}\t{description}\n"
    header = "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(header + tsv_line)
    else:
        with open(RESULTS_TSV, "a") as f:
            f.write(tsv_line)

    print(f"Recorded: {commit} | val_bpb={val_bpb:.6f} | {status} | {description}")
    return result


def recall_experiments(query: str, limit: int = 5) -> list:
    """Semantically search past experiments for relevant findings."""
    result = run_engram("recall", query=query, limit=limit)

    if "error" in result:
        print(f"Recall failed, falling back to results.tsv", file=sys.stderr)
        return _fallback_search(query)

    memories = result.get("memories", [])

    if not memories:
        print("No relevant prior experiments found.")
        return []

    print(f"\n{'='*60}")
    print(f"PRIOR KNOWLEDGE: {len(memories)} relevant experiments found")
    print(f"Query: \"{query}\"")
    print(f"{'='*60}")

    for i, mem in enumerate(memories):
        score = mem.get("score", 0)
        content = mem.get("content", "")
        meta = mem.get("metadata", {})
        print(f"\n[{i+1}] (relevance: {score:.2f})")
        print(f"    {content}")
        if meta.get("commit"):
            print(f"    commit: {meta['commit']} | val_bpb: {meta.get('val_bpb', '?')}")

    print(f"{'='*60}\n")
    return memories


def _fallback_search(query: str) -> list:
    """Simple keyword search in results.tsv when Engram is unavailable."""
    if not RESULTS_TSV.exists():
        return []
    results = []
    query_lower = query.lower()
    for line in RESULTS_TSV.read_text().strip().split("\n")[1:]:
        if query_lower in line.lower():
            results.append({"content": line, "score": 0.5})
    return results


def show_history():
    """Show all experiment results."""
    if not RESULTS_TSV.exists():
        print("No experiments recorded yet.")
        return

    print(RESULTS_TSV.read_text())


def show_stats():
    """Show experiment statistics."""
    if not RESULTS_TSV.exists():
        print("No experiments recorded yet.")
        return

    lines = RESULTS_TSV.read_text().strip().split("\n")[1:]  # skip header
    total = len(lines)
    keeps = sum(1 for l in lines if "\tkeep\t" in l)
    discards = sum(1 for l in lines if "\tdiscard\t" in l)
    crashes = sum(1 for l in lines if "\tcrash\t" in l)

    bpbs = []
    for line in lines:
        parts = line.split("\t")
        if len(parts) >= 2:
            try:
                bpb = float(parts[1])
                if bpb > 0:
                    bpbs.append(bpb)
            except ValueError:
                pass

    best_bpb = min(bpbs) if bpbs else None

    print(f"\n{'='*40}")
    print(f"EXPERIMENT STATISTICS")
    print(f"{'='*40}")
    print(f"Total experiments: {total}")
    print(f"Kept (improved):   {keeps}")
    print(f"Discarded:         {discards}")
    print(f"Crashed:           {crashes}")
    print(f"Hit rate:          {keeps/total*100:.1f}%" if total > 0 else "N/A")
    if best_bpb:
        print(f"Best val_bpb:      {best_bpb:.6f}")
    print(f"{'='*40}\n")


def main():
    parser = argparse.ArgumentParser(description="Research Memory — Engram for autoresearch")
    sub = parser.add_subparsers(dest="command")

    # remember
    rem = sub.add_parser("remember", help="Record an experiment result")
    rem.add_argument("--commit", required=True)
    rem.add_argument("--val-bpb", type=float, required=True)
    rem.add_argument("--vram-gb", type=float, required=True)
    rem.add_argument("--status", choices=["keep", "discard", "crash"], required=True)
    rem.add_argument("--description", required=True)
    rem.add_argument("--agent", default="agent-1")
    rem.add_argument("--reasoning", default="")
    rem.add_argument("--recall-context", default="")

    # recall
    rec = sub.add_parser("recall", help="Search prior experiments")
    rec.add_argument("query", type=str)
    rec.add_argument("--limit", type=int, default=5)

    # history
    sub.add_parser("history", help="Show all results")

    # stats
    sub.add_parser("stats", help="Show statistics")

    args = parser.parse_args()

    if args.command == "remember":
        remember_experiment(
            args.commit, args.val_bpb, args.vram_gb,
            args.status, args.description, args.agent,
            args.reasoning, args.recall_context,
        )
    elif args.command == "recall":
        recall_experiments(args.query, args.limit)
    elif args.command == "history":
        show_history()
    elif args.command == "stats":
        show_stats()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
