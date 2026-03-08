# The Karpathy Problem

> "Git(Hub) is *almost* but not really suited for [multi-agent collaborative research]. It has a softly built in assumption of one 'master' branch, which temporarily forks off into PRs just to merge back a bit later." — [@karpathy, March 2026](https://x.com/karpathy/status/2029701092347630069)

## The Problem

[Autoresearch](https://github.com/karpathy/autoresearch) proves that AI agents can autonomously improve ML models overnight. But Karpathy identified a critical limitation: **git can't scale this to multiple agents**.

Git assumes convergence — branches merge back to master. Multi-agent research is divergent — agents explore different directions simultaneously. They need to share *knowledge*, not merge *code*.

## Our Hypothesis

**Semantic memory ([Engram](https://github.com/terronexdev/engram)) can solve the multi-agent collaboration problem** that git alone cannot.

Instead of reading PRs and Discussions for "inspiration" (Karpathy's suggestion), agents use semantic recall to discover relevant prior experiments across all agents:

```python
# Before experimenting: What has anyone tried that's similar?
memories = recall("learning rate experiments on small models")
# → Returns semantically relevant findings from ANY agent, ANY branch

# After experimenting: Share findings
remember("Doubled LR from 0.04→0.08. val_bpb improved 0.997→0.991. Keep.",
         tags=["learning-rate", "keep"],
         metadata={"val_bpb": 0.991, "agent": "agent-3"})
```

## How It Works

```
┌─────────────────────────────────────────────┐
│              Git (unchanged)                │
│  Code versioning: train.py diffs, commits   │
│  Each agent on its own branch               │
└─────────────────────────────────────────────┘
                    +
┌─────────────────────────────────────────────┐
│         Engram (new knowledge layer)        │
│  Shared research-brain.engram file          │
│  Semantic search across ALL experiments     │
│  Cross-agent knowledge transfer             │
│  No merging required                        │
└─────────────────────────────────────────────┘
```

Git handles **what changed** (code). Engram handles **what was learned** (knowledge).

## Experimental Design

See [HYPOTHESIS.md](./HYPOTHESIS.md) and [METHOD.md](./METHOD.md) for full details.

- **Control**: Standard autoresearch (git + results.tsv)
- **Test**: Autoresearch + Engram semantic memory
- **Metric**: Experiments-to-improvement ratio, redundant experiment rate
- **Hardware**: NVIDIA RTX 4070 Super (12GB VRAM)
- **Live results**: [karpathy-problem.vercel.app](https://karpathy-problem.vercel.app) *(coming soon)*

## Project Structure

```
├── prepare.py           # Data prep (from autoresearch, unmodified)
├── train.py             # Model training (agent modifies this)
├── program.md           # Agent instructions
├── research_memory.py   # Engram integration layer (Python)
├── engram_cli.js        # Engram Trace bridge (Node.js)
├── HYPOTHESIS.md        # Scientific hypothesis
├── METHOD.md            # Experimental method
├── results.tsv          # Experiment log (backward compatible)
└── site/                # Vercel progress site
```

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Prepare data (one-time)
uv run prepare.py --num-shards 2

# 3. Run baseline
uv run train.py

# 4. Start autonomous research with Engram
# (See program.md for agent instructions)
```

## Status

🔬 **Phase 1: Infrastructure** — Setting up and adapting for RTX 4070 Super

## Credits

- [Andrej Karpathy](https://github.com/karpathy) — autoresearch concept and codebase
- [Terronex](https://github.com/terronexdev) — Engram semantic memory system
- Built with [@terronex/engram](https://www.npmjs.com/package/@terronex/engram) v2.1.2

## License

MIT
