# Experimental Protocol

## Study: The Karpathy Problem
**Date:** March 8, 2026
**Researcher:** Terronex (automated via Claw)
**Hardware:** NVIDIA GeForce RTX 4070 Super (12GB VRAM), WSL2/Ubuntu

## Objective
Test whether semantic memory (Engram v2.1) enables more efficient hyperparameter
discovery in autonomous research compared to flat file storage (results.tsv).

## Background
Karpathy observed that git's branch-and-merge model assumes convergence, but
multi-agent research needs divergent exploration with knowledge sharing. His
autoresearch framework uses `results.tsv` for experiment tracking. We test
whether replacing this with Engram's semantic memory improves outcomes.

## Independent Variable
- **Control:** Agents use results.tsv + simple heuristics to decide what to try
- **Test:** Agents use Engram semantic recall to inform decisions

## Dependent Variables
1. **Best val_bpb achieved** (primary) — lower is better
2. **Hit rate** — percentage of experiments that improve over current best
3. **Time to first improvement** — iterations until first keep
4. **Cumulative improvement** — total delta from baseline over time

## Controlled Variables
- Same hardware (RTX 4070 Super, single GPU)
- Same model architecture (11.5M param GPT, depth=4, head_dim=64)
- Same training data (FineWeb-Edu, 2 shards)
- Same training budget per iteration (300 seconds)
- Same hyperparameter search space (7 params, same ranges)
- Same random seed for model initialization
- Sequential execution (no parallel agent interference)

## Search Space
| Parameter | Range | Default | Type |
|-----------|-------|---------|------|
| EMBEDDING_LR | 0.1 - 2.0 | 0.6 | float |
| UNEMBEDDING_LR | 0.001 - 0.02 | 0.004 | float |
| MATRIX_LR | 0.005 - 0.2 | 0.04 | float |
| SCALAR_LR | 0.1 - 2.0 | 0.5 | float |
| WEIGHT_DECAY | 0.0 - 0.5 | 0.2 | float |
| WARMUP_RATIO | 0.0 - 0.15 | 0.0 | float |
| WARMDOWN_RATIO | 0.2 - 0.8 | 0.5 | float |

## Procedure

### Phase 1: Baseline (Complete)
1. Train model with default hyperparameters
2. Record val_bpb as baseline: **1.189370**
3. Record hardware metrics (VRAM, throughput, time)

### Phase 2: Control Group (In Progress)
1. Run orchestrator in `control` mode for 1 hour
2. Agent cycles through parameters, applies random perturbations
3. Decisions informed by experiment history in results.tsv
4. Keep changes that improve val_bpb, discard others
5. Record all results to experiments.jsonl

### Phase 3: Test Group (Pending)
1. Clear Engram memory (fresh start, same conditions as control)
2. Run orchestrator in `test` mode for 1 hour
3. Agent queries Engram before each decision
4. Semantic recall provides context on prior experiments
5. Agent uses recall insights to choose direction and magnitude
6. Record all results to experiments.jsonl + Engram

### Phase 4: Analysis
1. Compare best val_bpb between groups
2. Compare hit rates (improvements / total experiments)
3. Plot improvement curves over iterations
4. Statistical significance test if sample sizes permit
5. Qualitative analysis of decision quality

## Limitations
- Single run per group (no statistical power without repetition)
- Scaled-down model (11.5M vs 100M+ in original autoresearch)
- Simple heuristic agent (not LLM-driven as in full autoresearch)
- Control agent uses random perturbation, not pure random search
- 1-hour duration limits number of experiments (~8-9 per group)

## Ethical Considerations
- No human subjects
- Open source code and data
- Results published regardless of outcome

## Data Storage
- `results/control_*/experiments.jsonl` — control group raw data
- `results/test_*/experiments.jsonl` — test group raw data
- `results/*/summary.json` — per-run summaries
- `results/*/iter_*.log` — full training logs per iteration
- `~/.karpathy-problem/research-brain.engram` — Engram memory file
