# The Karpathy Problem — Hypothesis

## Problem Statement

Andrej Karpathy identified a fundamental limitation in using git for multi-agent autonomous research collaboration (March 2026):

> "Git(Hub) is *almost* but not really suited for this. It has a softly built in assumption of one 'master' branch, which temporarily forks off into PRs just to merge back a bit later."

The core issue: git's data model assumes **convergence** (branches → merge → master). Multi-agent research is **divergent** — agents explore different directions simultaneously and need to share findings without merging code.

## Hypothesis

**Semantic memory (Engram) enables more efficient multi-agent autonomous research than git's branch model** by:

1. Eliminating redundant experiments through semantic recall of prior findings
2. Enabling cross-agent knowledge transfer without branch merging
3. Providing natural-language discovery of related experiments across all agents

## Experimental Design

### Control Group: Standard Autoresearch
- Git + results.tsv (Karpathy's original design)
- Agents work independently on separate branches
- No cross-agent knowledge sharing
- Knowledge discovery: linear scan of results.tsv

### Test Group: Autoresearch + Engram
- Git for code versioning (train.py diffs) — unchanged
- Engram file as shared knowledge store (replaces results.tsv for discovery)
- Before each experiment: `recall()` for similar past experiments
- After each experiment: `remember()` findings with semantic embeddings
- Cross-agent knowledge transfer via shared Engram file

### Metrics
1. **Experiments-to-improvement ratio**: How many experiments needed to find an improvement?
2. **Redundant experiment rate**: How often does an agent try something already proven ineffective?
3. **Best val_bpb achieved in N hours**: Final model quality
4. **Knowledge transfer events**: Times an agent's decision was influenced by another agent's findings

### Hardware
- **GPU**: NVIDIA GeForce RTX 4070 Super (12GB VRAM)
- **Platform**: WSL2 on Windows, Ubuntu
- **Note**: Results are platform-specific per Karpathy's design (fixed 5-min time budget)

### Software
- **Research Agent LLM**: Claude Sonnet (via API)
- **Training Code**: Fork of karpathy/autoresearch
- **Knowledge Layer**: @terronex/engram v2.1.2 + @terronex/engram-trace
- **Embeddings**: all-MiniLM-L6-v2 (384 dims, local)

## Predictions

1. Engram-enabled agents will achieve the same val_bpb improvement in ~30% fewer experiments
2. Redundant experiment rate will drop from ~15-20% to <5%
3. Multi-agent runs with Engram will outperform multi-agent runs without Engram
4. The knowledge graph will reveal emergent clustering of successful strategies

## References

- Karpathy tweet: https://x.com/karpathy/status/2029701092347630069
- Autoresearch repo: https://github.com/karpathy/autoresearch
- Engram: https://github.com/terronexdev/engram
- Engram Trace: @terronex/engram-trace (NPM)
