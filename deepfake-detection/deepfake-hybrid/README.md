# Cross-dataset deepfake detection: GenD + ForensicsAdapter hybrid

Image-level deepfake detection evaluated in a **cross-dataset** setting: trained on FaceForensics++ (FF++), tested on DFDC.
The final model is a score-level hybrid of two published detectors; all other combinations we tried are reported as negative results below.

## Final result (FF++ -> DFDC, 10,000 test frames, 4,355 videos)

| Model | Video AUROC | Video acc. | Frame AUROC | Frame acc. |
|---|---|---|---|---|
| GenD (frozen CLIP features + linear head) | 82.4% | 74.2% | 79.9% | 72.4% |
| ForensicsAdapter (authors' released weights) | 83.0% | 74.8% | 80.8% | 73.2% |
| **Hybrid (mean of the two fake probabilities)** | **84.9%** | **76.2%** | **82.6%** | **74.8%** |

Published reference (video-level AUROC, FF++ -> DFDC): GenD 87.1%, ForensicsAdapter 87.2% (full training, ~32 frames per video).

## Data and protocol
* **Train:** FF++ (GenD's preprocessed release, x1.3 face crops). 12,000 real + 12,000 fake frames (fakes sampled from Deepfakes, Face2Face, FaceSwap, NeuralTextures), each in 4 views (original, horizontal flip, 2 augmented copies) = 96,000 rows for the linear head.
* **Test:** DFDC (GenD's preprocessed release). 5,000 real + 5,000 fake frames, random sample (fixed seed 0), from 4,355 videos (~2.3 frames per video). Never used for training or tuning.
* **No tuning on DFDC.** Head regularisation (C = 0.01), the equal 50/50 fusion weights and the 0.5 threshold were fixed beforehand. DFDC labels are used only for scoring.
* The GenD and ForensicsAdapter backbones were trained by their authors on FF++; we did not retrain them.
* The datasets are **not included** (FF++ and DFDC have their own terms of use).

## Limitations (read before citing the numbers)
* The test frames are a random sample, not a predefined split, and only ~2.3 frames per video are averaged for the video-level scores, so the video-level numbers are weaker than a 32-frames-per-video protocol would give.
* The three components saw slightly different FF++ preprocessing (GenD's crops; DeepfakeBench-style data for ForensicsAdapter).
* In-dataset (FF++ hold-out) accuracies of 93-96% from `experiments.py` use a video-level split that is **not person-disjoint** and are optimistic. `indataset.py` implements a person-disjoint split but has **not been run yet**.
* Absolute accuracy at a fixed 0.5 threshold is below the published AUROC levels; 85% cross-dataset accuracy was not reached.

## Negative results (same train/test protocol)
* GenD + MoE-FFD feature concatenation: 78.5-79.0% video AUROC, lower than GenD alone (it overfits FF++-specific cues: in-dataset accuracy rises, cross-dataset falls).
* Flip/augmented training copies: about +0.3 points. Test-domain feature standardisation: no change.
* Unsupervised self-training on unlabeled DFDC (`adapt.py`): 82.4% -> 82.6% video AUROC; a perfect-threshold diagnostic shows calibration is not the bottleneck.
* An earlier CaiT experiment (full fine-tune 68.3% AUROC, LayerNorm-only 64.6%) was done outside this repo.

## Files and run order
1. `extract_features.py` - GPU, once. Saves `feats/*.npz` (GenD and MoE-FFD features, flip/augmented views).
2. `experiments.py` - CPU. GenD vs GenD+MoE-FFD, views, standardisation, linear vs MLP head.
3. `adapt.py` - CPU. Self-training diagnostic.
4. `indataset.py` - CPU. Person-disjoint FF++ hold-out (not yet run).
5. `score_fa.py` - GPU. Scores DFDC frames with ForensicsAdapter (needs its repo and released weights).
6. `fuse.py` - CPU. Fuses the GenD head scores and ForensicsAdapter scores; prints the table above.
7. `hybrid_detector.py` - packages both detectors in one `HybridDetector` module (`predict(path)`).

8. `make_figures.py` - results bar chart (`figures/results_summary.png`).
9. `show_predictions.py` - grid of random real/fake DFDC frames with the hybrid's verdicts (run in the session where the model is loaded).

External code and weights: [GenD](https://github.com/yermandy/GenD) (`yermandy/GenD_CLIP_L_14`), [ForensicsAdapter](https://github.com/OUC-VAS/ForensicsAdapter) (CC BY-NC 4.0), [MoE-FFD](https://github.com/LoveSiameseCat/MoE-FFD).

## Papers
* Yermakov et al., *Deepfake Detection that Generalizes Across Benchmarks*, WACV 2026 - https://arxiv.org/abs/2508.06248
* Cui et al., *Forensics Adapter: Adapting CLIP for Generalizable Face Forgery Detection*, CVPR 2025 - https://arxiv.org/abs/2411.19715
* *MoE-FFD: Mixture of Experts for Generalized and Parameter-Efficient Face Forgery Detection*, IEEE TDSC 2025 - https://arxiv.org/abs/2404.08452
