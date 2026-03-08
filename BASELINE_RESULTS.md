# Baseline Training Results

## Session: cool-nudibranch
**Date:** 2026-03-08 18:48 EDT
**Hardware:** NVIDIA GeForce RTX 4070 Super (12GB VRAM)

## Configuration
- **Model Size:** 11.5M parameters  
- **Architecture:** Depth 4, Head dim 64 (scaled from original 8/128)
- **Batch Size:** Device=16, Total=131,072
- **Training Budget:** 300 seconds (5 minutes)

## Results
- **Final val_bpb:** 1.189370
- **Training Time:** 300.1 seconds
- **Total Time:** 339.7 seconds (including validation)
- **Peak VRAM:** 1,933.6 MB (16.1% utilization)
- **MFU:** 4.57% 
- **Total Tokens:** 271.1M
- **Steps:** 2,068

## Performance Notes
- Training completed without errors
- VRAM usage well within hardware limits
- Consistent ~890K tokens/sec throughput
- Loss decreased from 9.01 → 3.32 (training), final validation BPB: 1.189

## Next Phase
This baseline establishes the performance of standard autoresearch (git-based collaboration) on our hardware. Next: implement Engram semantic memory integration and compare results.