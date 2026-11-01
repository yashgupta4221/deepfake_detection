# FS-VFM experiments protocol (written 2026-10-19, BEFORE any FS-VFM run; do not edit after seeing results)

## Arms (all: train FF++ c23 only; test DFDC and Celeb-DF v2; same crops/manifests/metrics as Stage B)
1. OSDFD (CLIP ViT-L/14, frozen + LoRA/CDC/FSM/SCL), 2,000 its, seeds 0-2 -- ALREADY RUN (Stage B).
2. FS-VFM ViT-B/16, official full fine-tuning recipe (10 epochs), seeds 0-2 -- baseline reproduction (cells F1-F7).
3. OSDFD-FSVFM: the OSDFD components on the frozen FS-VFM ViT-B/16 backbone (`fsvfm_osdfd.py`), 2,000 its, seeds 0-2.
   ~1.16M trainable parameters (same order as OSDFD's ~1.5M). Recipe = OSDFD's (Adam, constant lr 1e-4, batch 24 real + 6 per forgery,
   FSM p=0.5, lambda_SCL=1), exactly as used for OSDFD in Stage B. No augmentation.

## No tuning on test data
Arm 3 is ONE fixed configuration chosen before running (above). No hyper-parameter, layer choice or checkpoint is selected using DFDC or
Celeb-DF. Last checkpoint only. If a bug is found and fixed, every arm-3 seed is re-run with the fixed code and the fix is reported.

## Reproduction check (arm 2)
If the 3-seed mean frame AUROC of arm 2 is more than 0.03 below the paper's ViT-B figures (Celeb-DF v2 0.8613, DFDC 0.8184), treat the
baseline as not reproduced, investigate (data/crop differences) and say so before using arm 2 in any comparison.

## Verdict rule (same as round 2)
Arm A counts as better (worse) than arm B only if its mean differs by more than the larger of the two seed-stds on ALL FOUR metrics
(DFDC frame/video, Celeb-DF frame/video AUROC). Otherwise "no reliable difference". Report mean +- sample std and all per-seed values.
Comparisons: arm 3 vs arm 1 (does the backbone change help OSDFD?) and arm 3 vs arm 2 (do the OSDFD components help FS-VFM?).

## Known limits to state in any write-up
n = 3 seeds; two test sets; arm 2 trains 10 epochs on our frame subset (not the official 32 frames/video), arm 3 only 2,000 iterations;
crops differ slightly from the FS-VFM paper's preprocessing; FS-VFM is an arXiv extension (journal status unconfirmed).
