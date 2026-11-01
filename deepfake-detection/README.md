# Image-only deepfake detection - cross-dataset study

Research project on image-only deepfake detection for biometric authentication. Strict protocol: **train on FaceForensics++ (c23) only; test on Celeb-DF v2 and DFDC** (never used for training or tuning), plus an in-domain held-out FF++ split. Results are reported as measured, with the seed spread; nothing was tuned on a test set.

## Main result (fdnet, frozen CLIP ViT-L/14 + LoRA; AUROC, 3 seeds, mean +- sample std)
| | Celeb-DF v2 frame / video | DFDC (all 40k frames) frame / video | FF++ held-out frame |
|---|---|---|---|
| A0 - plain baseline | 0.814 +- 0.022 / 0.913 +- 0.014 | 0.809 +- 0.007 / 0.831 +- 0.007 | 0.973 |
| **C1 - + augmentation + flip, cosine schedule** | **0.878 +- 0.010** / 0.927 +- 0.010 | **0.818 +- 0.002 / 0.844 +- 0.003** | 0.961 |

Honest reading: C1 improves clearly over the baseline on Celeb-DF frame AUROC and on DFDC, and is much more stable across seeds; under the strict pre-registered rule (gain larger than the larger seed std on all four metrics) it passes 3 of 4 (Celeb-DF video misses by 0.0007), so the formal verdict is "no reliable difference". Against our OSDFD reproduction it is above on DFDC and level on Celeb-DF. DFDC AUROC is about 0.82; in-domain FF++ drops slightly with augmentation. Arms C2 (self-blended fakes) and C3 (multi-layer fusion) were lost in a session reset and **not re-run** (see `osdfd-repro/PROTOCOL_NEW.md`, Addendum F). Details: `results/fdnet_C1_vs_A0_results.xlsx`.

## Layout
- `osdfd-repro/` - our detector `fdnet.py` (+ tests), the OSDFD reproduction, OSDFD x MoE-FFD / FS-VFM hybrids, protocols (`PROTOCOL*.md`, written before the runs; Addenda A-F), Kaggle notebook (`fdnet_v2_kaggle.ipynb`) and run scripts (`kaggle_runs/`).
- `deepfake-hybrid/` - earlier GenD + ForensicsAdapter feature-level hybrid.
- `wavie-repro/` - WAVIE reproduction.
- `results/` - final results spreadsheet.

## Notes
- Datasets are not included (FaceForensics++, Celeb-DF v2, DFDC have their own licences); model checkpoints are not committed.
- OSDFD (Kong et al.) and MoE-FFD (Kong et al.) are re-implemented from the papers; no third-party code is included.
- Run order and exact commands: `osdfd-repro/NOTEBOOK_CELLS*.md`, `osdfd-repro/fdnet_v2_kaggle.ipynb`.
