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
- **Live results**: [https://karpathy-problem-site.vercel.app](https://karpathy-problem-site.vercel.app)

## Conclusion (March 10, 2026)

Our hypothesis — that Engram semantic memory would provide a significant advantage for multi-agent
collaboration in this specific hyperparameter search problem — was **not supported**.

-   **Heuristic agents (R2 Control) found the best overall score:** The simplest approach (TSV + Heuristic)
    achieved the deepest minimum (val_bpb = 1.184903). In small, bounded search spaces,
    random exploration (even with weak heuristics) can sometimes find impactful solutions by chance.
-   **LLM agents provided consistent exploration, but not deeper minima:** Both LLM-powered groups
    (R3 Control & R3 Test) showed more even distributions of improvements, preventing long plateaus.
    However, they did not surpass the best score found by the heuristic. This indicates the LLM's
    reasoning was either too conservative or the search space too simple for complex inference.
-   **Memory format (TSV vs. Engram) showed no significant advantage:** Across both heuristic and LLM
    reasoning, the choice of memory format did not produce a substantial difference in the final best val_bpb.

This experiment highlights that the true value of semantic memory and advanced reasoning will likely
emerge in more complex, unstructured search spaces (e.g., code-level changes), where keyword-based search
fundamentally breaks down. This aligns with Karpathy's own observations about code-level modifications.

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

✅ **Complete** — All experiments finished March 10, 2026. Results published at [karpathy-problem-site.vercel.app](https://karpathy-problem-site.vercel.app)

## Credits

- [Andrej Karpathy](https://github.com/karpathy) — autoresearch concept and codebase
- [Terronex](https://github.com/terronexdev) — Engram semantic memory system
- Built with [@terronex/engram](https://www.npmjs.com/package/@terronex/engram) v2.1.2

## License

MIT
