# WAVIE reproduction

**Two modes.** (1) Paper protocol: FF++ raw -> Celeb-DF v1/v2, WildDeepFake (needs data not included). (2) **Adaptation used here:** same architecture, loss, optimiser and schedule, but trained on the existing GenD pre-cropped FF++ frames (x1.3 crops) and tested on DFDC. Mode 2 is NOT the paper's data protocol; label its numbers "WAVIE architecture and training recipe, FF++ (GenD crops) -> DFDC" and do not compare them directly with the paper's Celeb-DF/WDF numbers.

Reimplementation of **WAVIE: Wavelet Augmented Vision Intermediate Embeddings** (Pulkit, Ruhela, Kapoor, Bhavsar; arXiv 2609.14437),
written from the paper's Section III-IV. Not affiliated with the authors; cite the paper. The authors' code is at
github.com/Kappuccino111/WAVIE-Master (not used here, only suitable for cross-checking).

## What is reproduced
* Frozen CLIP ViT-L/14, [CLS] token of each of the 24 blocks (width 1024, before projection) -> shared MLP Q1 -> learned per-layer,
  per-dimension softmax aggregation -> 3-level db6 DWT (low band = 137) -> linear 137x137 on the low band only -> IDWT -> Q2 -> linear classifier.
* Loss BCE + 0.1 SupCon (tau 0.07); AdamW lr 1e-3, wd 1e-2, batch 32, AMP, OneCycleLR, EMA 0.999, 20 epochs, fixed seeds.
* Trainable parameters: **2,143,707** (paper: ~2.14M). Verified by `python -c "from model import WAVIE; ..."`.
* DWT/IDWT are built as fixed linear maps from `pywt`, so they equal `pywt.wavedec/waverec` and are differentiable (perfect reconstruction checked).
* Ablation switches matching Table III: `--mode no_fusion | no_wavelet | avg_agg`.

## Data protocol (paper, Sec. IV-A) - you must supply the data
* Train: FaceForensics++ **raw (c0)**, Deepfakes/Face2Face/FaceSwap/NeuralTextures + original youtube videos; real at 1 fps, fake at 4 fps;
  MTCNN largest face, +20 px, resize 224; paper reports 131,155 real / 175,465 fake frames, random 80/20 train/val split.
* Test (frame-level AUROC): Celeb-DF v1 (8,073 real / 16,302 fake), Celeb-DF v2 (10,165 / 21,458), WildDeepFake (58,659 / 107,003).
* None of these datasets are included. Celeb-DF and FF++ require access requests; check each project page for terms.

## Run order - adaptation on existing data (FF++ GenD crops -> DFDC)
0. (Archive route, saves disk) `python extract_subset.py --archive <FF...tar.lz4> --kind ffpp --out crops/ffpp` and `... --kind dfdc --n_test 20000 --out crops/dfdc` - streams the archive and writes only the sampled frames plus `manifest.csv`; then skip step 1 below.
1. (Folder route, if already unpacked) `python make_manifest.py --ff_root <.../FF-x1.3-th0.5-all> --dfdc_root <.../DFDC-pre-x1.3-th0.5-all> --out crops`
   (training make-up mimics the paper: fake:real = 1.34 with the four methods equal; test = up to 20,000 random frames per class from DFDC, seed 0)
2. `python extract_clip_cls.py --manifest crops/ffpp/manifest.csv --out feats/ffpp` and the same for `crops/dfdc`
3. `python train.py --feats feats/ffpp --out runs/wavie` (paper-style random frame split) and again with `--split video --out runs/wavie_vsplit` (no leakage)
4. `python eval.py --ckpt runs/wavie/wavie_ema_last.pt --sets dfdc=feats/dfdc` (frame-level AUROC = primary; per-video mean score reported as an extra)
Differences from the paper in this mode: no MTCNN step (crops already made, x1.3 scale instead of +20 px), FF++ compression level of the GenD release is not the raw c0 videos, 20,000-frame DFDC test instead of Celeb-DF/WDF. Do not tune anything on DFDC.

## Run order - paper protocol
1. `python preprocess.py ffpp --root <FF++ root> --out crops/ffpp`
2. `python preprocess.py videos --real <Celeb-real dirs> --fake <Celeb-synthesis dir> --fps 1 --out crops/cdf2` (same for v1); `python preprocess.py images ...` for WDF.
3. `python extract_clip_cls.py --manifest crops/<set>/manifest.csv --out feats/<set>` (GPU, once per set; ~49 KB per frame in fp16).
4. `python train.py --feats feats/ffpp --out runs/wavie` (add `--in_ram` if memory allows).
5. `python eval.py --ckpt runs/wavie/wavie_ema_last.pt --sets cdf1=feats/cdf1 cdf2=feats/cdf2 wdf=feats/wdf`

Paper targets (for comparison only, never for tuning): FF++ 0.980, CDF-v1 0.8518, CDF-v2 0.8520, WDF 0.8313 (frame-level AUROC).

## Where the paper is ambiguous (choices made here; report these if results differ)
| Item | Choice |
|---|---|
| SupCon representation | text says "after Q2", the equation uses the IDWT output; default Q2 output, `--supcon_on idwt` switches |
| "Expand box by 20 px (when in-bounds)" | expanded box used only if fully inside the image, else the plain box |
| Resize interpolation, PNG vs JPEG | unspecified; INTER_AREA, PNG |
| Test-set frame sampling | unspecified beyond "same preprocessing"; fps default 1, so frame counts will not match the paper's exactly |
| Epoch used for reporting | unspecified; final-epoch EMA weights, nothing chosen on test sets |
| Train/val split | frame-level random 80/20 as described; this leaks near-identical frames of one video across the split, so the FF++ number is optimistic |
| Table III ablation internals | "no fusion" = Q1(last-block CLS); "no wavelet" bypasses DWT, low-band layer and IDWT but keeps Q2 |

## Tests so far
Mock-feature smoke test only: shapes, parameter count, DWT perfect reconstruction, identity no-op of the wavelet block, train/eval plumbing.
No result on real data exists yet.
