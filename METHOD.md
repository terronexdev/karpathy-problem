# The Karpathy Problem — Method

## Phase 1: Infrastructure (Day 1)
- [x] Fork karpathy/autoresearch
- [x] Read and understand full codebase
- [ ] Adapt train.py for RTX 4070 Super (12GB VRAM)
- [ ] Run data preparation (prepare.py)
- [ ] Establish baseline val_bpb on our hardware
- [ ] Build research_memory.py (Engram integration layer)
- [ ] Create Vercel site for live progress tracking
- [ ] Set up GitHub repo (terronexdev/karpathy-problem)

## Phase 2: Control Group (Day 2)
- [ ] Run standard autoresearch (no Engram) for 8 hours
- [ ] Agent uses git + results.tsv only
- [ ] Log all experiments, decisions, and outcomes
- [ ] Record baseline metrics

## Phase 3: Engram Integration (Day 3)
- [ ] Add Engram recall/remember to experiment loop
- [ ] Run Engram-enabled autoresearch for 8 hours
- [ ] Same agent, same starting point, same hardware
- [ ] Compare metrics against control

## Phase 4: Multi-Agent (Day 4-5)
- [ ] Spawn 2-3 parallel agents
- [ ] Control: separate branches, no knowledge sharing
- [ ] Test: shared research.engram file
- [ ] Measure cross-agent knowledge transfer
- [ ] Document findings

## Phase 5: Publication (Day 6-7)
- [ ] Compile results
- [ ] Update Vercel site with final analysis
- [ ] Write summary for X/Twitter
- [ ] Open Discussion on karpathy/autoresearch with findings
- [ ] Link from Engram README

## Adaptations for RTX 4070 Super

### VRAM Budget: 12GB (vs 80GB H100)

| Parameter | H100 (Original) | RTX 4070 Super |
|-----------|-----------------|----------------|
| Depth | 8 layers | 4 layers |
| Model dim | 512 (8*64) | 256 (4*64) |
| Batch size | 128 | 16-32 |
| Total batch | 524K tokens | 65K-131K tokens |
| Params | ~50M | ~5-10M |

### Flash Attention
- H100 uses varunneal/flash-attention-3 (Hopper-native)
- 4070 Super uses kernels-community/flash-attn3 (Ada fallback)
- If FA3 fails: fall back to PyTorch SDPA

### What This Changes
- Smaller model = fewer parameters = faster experiments
- val_bpb numbers won't match H100 results (expected and fine)
- The ENGRAM COMPARISON is still valid — same hardware for both groups
- Fixed 5-min time budget still applies

## Data Collection

Every experiment logs:
1. Git commit hash
2. val_bpb (primary metric)
3. Peak VRAM (GB)
4. Status (keep/discard/crash)
5. Description of what was tried
6. **Engram recall results** (what prior knowledge influenced the decision)
7. **Agent reasoning** (why this experiment was chosen)

All data published to the Vercel site in real-time.
