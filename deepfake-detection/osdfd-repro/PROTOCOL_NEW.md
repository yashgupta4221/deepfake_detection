# PROTOCOL_NEW - fdnet (written BEFORE any fdnet result exists; do not edit after the first run)

Goal: an image-only deepfake detector, trained on FaceForensics++ c23 only, tested cross-dataset (DFDC, Celeb-DF v2). Genuine numbers only.

## Method (all in fdnet.py; built on frozen CLIP ViT-L/14 + LoRA, which must be cited: Radford 2021, Hu 2022)
Arms (4 training configurations, 3 data-order seeds each, shared init_seed=0 so weights can be merged):
- A0 baseline:      --fuse last  --grp none   (plain LoRA + BCE)
- A1 +fusion:       --fuse multi --grp none   (layers 8,12,16,20,24)
- A2 +worst-group:  --fuse last  --grp dro    (groups real/DF/F2F/FS/NT, eta 0.01, floor 0.02)
- A3 both:          --fuse multi --grp dro
Post-hoc (free, no extra training): for each arm merge its 3 seeds by weight averaging (soup) or probability averaging (ens).
Fixed, NOT tuned: r=8, alpha=16, lr=1e-4 constant, 2000 iterations, batch 24 real + 6 per forgery, no augmentation, last checkpoint only.

## Data / partitions
Train: FF++ c23 manifest (same as all earlier work, video-level 90/10 train/val split, val used only as a sanity print).
DFDC test manifest is split by md5(video id) into DEV (~1/3 of videos) and FINAL (~2/3). Celeb-DF v2 is FINAL only.
DEV is used ONLY to choose the configuration. FINAL sets are looked at only after the choice is written down.

## Selection rule (decided now)
Score(arm) = mean over its 3 seeds of the average of DFDC-DEV frame AUROC and DFDC-DEV video AUROC.
Proposed configuration = highest Score; if the top arm beats a simpler arm by less than the larger seed-std, take the simpler arm.
Same rule chooses between none / soup / ens for the chosen arm.

## Reporting (everything is reported, nothing dropped)
Table: all 4 arms x {single seed mean+-std, soup, ens} on Celeb-DF v2 and DFDC-FINAL (frame AUROC, video AUROC, acc@0.5).
Full-DFDC numbers (dev+final) may be shown only as a secondary column labelled "includes the selection half".
A component "helps" only if its mean beats A0's mean by more than the larger of the two seed-stds on ALL FOUR final metrics
(Celeb-DF frame/video, DFDC-FINAL frame/video); "hurts" symmetric; otherwise "no reliable difference".
Comparisons: (1) controlled, same code: A0 vs components; (2) published numbers quoted as published and labelled as different protocol;
(3) our own OSDFD reproduction (3 seeds, full DFDC / Celeb-DF v2) labelled as such. No claim of "state of the art" unless a same-protocol run supports it.

## Known limits (state in the paper)
Frame subset of FF++ is small (about 7k real + 4x2.1k fake training frames), DFDC test is a 20k-frame / 4,996-video subset, one backbone, 3 seeds.
Novelty must be checked against R2M (model merging for deepfake detection, arXiv 2509.24367) and multi-layer-fusion deepfake papers before any claim.

## Addendum A (written 2026-10-19, before any fdnet result exists): compute budget and partial-seed rule
Jobs run in ROUNDS: round s = seed s for all four arms (A0..A3). Two GPUs may run two jobs in parallel; this changes nothing else.
If time runs out, only completed rounds count. An arm with fewer than 3 seeds is reported with its n and mean, but NO "helps/hurts" verdict is made
from fewer than 3 seeds (verdict = "not enough seeds"). The DEV-based selection then uses the seeds that exist for every arm, and the paper says so.

## Addendum B (written 2026-10-19, before any fdnet result exists): in-domain FF++ test set
Requested: also report FF++. The FF++ test set is the 10% of FF++ videos held out from training by the video-level split (split_seed 0), scored with the
LAST checkpoint; it is NOT the official FF++ test list and must be described that way. The 'FF++ val AUC' printed during training is the same data and was only
printed (never used to pick a checkpoint or a configuration). It is reported as an in-domain reference and is NOT used for the selection rule
(which still uses DFDC-DEV only). Note: in-domain AUROC close to 1.0 does not imply cross-dataset quality.

## Addendum C (written 2026-10-24, before ANY fdnet result exists - the first Kaggle session ended before a single evaluation finished): v2 recipe arms and budget
Motivation: the 2000-iteration, no-augmentation recipe (A0-A3) is short and clean, but DFDC contains heavy compression and varied capture conditions, so a
stronger training recipe is the most direct route to better cross-dataset AUROC. v2 adds switches that are all OFF by default (the old arms reproduce v1
bit-for-bit; verified by test_fdnet_v2.py): --sched cos (warm-up + cosine decay), --aug std (random crop-resize, colour jitter, blur, JPEG; after any blending),
--flip, --sbi N (N self-blended pseudo-fakes per batch, made on the fly from real FF++ training frames; our landmark-free re-implementation of the idea of
Shiohara & Yamasaki, CVPR 2022 - NOT their code, and a 6th DRO group when --grp dro), --tta flip (evaluation only), --resume (crash insurance).
Peak lr 1e-4, warm-up 100 its, batch 24 real + 6 per forgery (+12 SBI in C2/C3), seed 0, init_seed 0, LAST checkpoint only, nothing tuned.

Stage 1 (seed 0, single run per arm; evaluated ONLY on DFDC-DEV and the FF++ held-out set - no Celeb-DF, no DFDC-FINAL):
  A0  original recipe (2000 its, constant lr)            - baseline, also seeds 1 (and later 2) for the final verdict
  C0  A0 + cosine schedule, 6000 its
  C1  C0 + --aug std --flip
  C2  C1 + --sbi 12
  C3  C2 + --fuse multi
Selection ladder (decided now): score = mean of DFDC-DEV frame and video AUROC. best = C0; for C1, C2, C3 in that order: if score(Ci) - score(best) >= 0.005 then best = Ci.
Flip-TTA is used in the final evaluation iff it raises the chosen arm's DEV score by >= 0.005. (Both with and without TTA are reported.)

Stage 2 (after the choice is written down): the chosen arm with seeds 0,1,2 (shared init_seed 0) and A0 with seeds 0,1,2; evaluate on Celeb-DF v2, DFDC-FINAL,
FF++ held-out; report mean +- std, every per-seed value, soup and ensemble of the 3 seeds. Verdict vs A0 uses the same rule as before (difference larger than the larger
seed-std on ALL FOUR final metrics, otherwise "no reliable difference"). The Stage-1 ladder is a single-seed comparison and is reported as such (DEV-only, no verdicts).
Leftover budget may fund extra arms (e.g. A1-A3 at 2000 its, or DRO on top of the chosen arm); each extra arm is added here BEFORE it is run.

Limits to state in the paper: ladder selection uses one seed (noisy; margin 0.005 is a guard, not a significance test); the self-blend is an approximation without
facial landmarks and may behave differently from the published method; augmentation strengths are untuned defaults; any run that was resumed is flagged
(DONE.json "resumed": true) and had a different data order after the restart.

## Addendum D (written 2026-10-25 00:30 IST, after Stage-1 results for A0/C0/C1 exist but BEFORE any Celeb-DF or DFDC-FINAL number of fdnet exists): comparison with our OSDFD reproduction
Goal stated by the author: show that the final detector is at least as good as OSDFD alone. Comparison is made against OUR OSDFD reproduction (CLIP ViT-L/14, 3 seeds, same crops, same DFDC 40,000-frame / 4,996-video test manifest and
Celeb-DF v2 518-video manifest), not against numbers quoted from papers. Metrics: DFDC full set (DEV+FINAL together, because the OSDFD numbers were computed on the full set; labelled
"includes the selection half") frame and video AUROC, and Celeb-DF v2 frame and video AUROC. Rule (same as the earlier hybrid-vs-OSDFD rule): our detector "beats" OSDFD only if its 3-seed mean is higher by MORE than the larger of
the two seed-stds on ALL FOUR metrics; "worse" symmetrically; "mixed" if it wins on some metrics and loses on others (reported as such, per metric); otherwise "no reliable difference".
Also reported, whatever the outcome: trainable parameters (ours about 1.35M vs OSDFD 2.89M) and training iterations (ours 2,000-6,000 vs OSDFD's shortened schedule in our reproduction).
Stage 2 therefore must also evaluate the chosen arm and A0 on the FULL DFDC manifest (set "dfdc_all", no partition) in addition to DFDC-FINAL.

## Addendum E (written 2026-10-25 01:00 IST, before any fdnet number exists on these sets): two extra TEST-ONLY datasets from the Kaggle DeepFakeBench upload
Source: Kaggle dataset "deepfakebench" (uploader jamestashvik), a re-upload of DeepFakeBench-preprocessed face frames (256x256 PNG); NOT the original releases; provenance/licence of the upload not verified. Chosen because these sets
are common in the papers we compare with (DFDC Preview is tested by OSDFD, MoE-FFD and FS-VFM; Celeb-DF v1 is a standard Celeb-DF variant), NOT because of any expected result. Used for testing only: never for training, never for selecting arms.
Not used: the "DFDC" part of that upload (all 132,116 frames are labelled FAKE, so AUROC is undefined), Celeb-DF v2 and FF++ (we use the official releases), UADFV (98 videos, too small).
Manifests (built by the cell in the chat, fixed sampling = 10 evenly spaced frames per video, sorted by file name):
  dfdcp      : ALL 5,047 videos, 49,023 frames (real 1,052 videos / 9,937 frames; fake 3,995 videos / 39,086 frames). Not the official DFDCP test split (the split file is not in the upload) - state this.
  cdf1_test  : the official Celeb-DF v1 test list, 100 videos, 1,000 frames (real 8 videos / 80 frames; fake 92 videos / 920 frames). Only 8 real videos: video-level AUROC on this set is very unstable and must be reported with that warning.
  cdf1_all   : (optional, secondary) all 1,203 Celeb-DF v1 videos, 10 frames each, labelled "all videos, not the official split".
Known confound (state in the paper): the crop style of this upload differs from our own x1.3 crops (DFDCP faces are cropped tighter than ours; Celeb-DF v1 crops include more of the shoulders), so a drop on these sets can partly be a crop-domain
effect and not only a generalisation effect. Evaluated with the same single code path (fdnet.py eval), 3 seeds for the chosen arm and for A0, reported as mean +- std with every per-seed value; no verdict on these sets is derived from fewer than 3 seeds.
These sets do NOT enter the OSDFD "beats / mixed" rule of Addendum D (OSDFD was not run on them); they are reported as additional results only.

## Addendum F (written 2026-10-25 16:30 IST, after the Stage-1 DEV results and the first final evaluation of A0_s0, A0_s1, C1_s0 were seen; before any additional seed was run)
Deviation from Addendum C: the arms C2 and C3 (and their DEV results) were lost when the Kaggle session was reset; they were NOT re-run. The ladder therefore could not be completed. The arm carried forward is **C1** (cosine schedule + --aug std + --flip, no SBI, no multi-layer fusion). It was chosen on DFDC-DEV only (C1 DEV score 0.8247 vs C0 0.8074, A0 0.8059/0.8183); flip-TTA is NOT used (its DEV gain for C1 was +0.0007 < 0.005). The paper must state that C2/C3 were not evaluated.
Disclosure: the final-set numbers of the three runs A0_s0, A0_s1, C1_s0 had been seen when this addendum was written. Stage 2 therefore adds seeds C1_s1, C1_s2, A0_s2 (init_seed 0, same recipe) and reports all three seeds per arm; the verdict rule of Addendum C/D (difference larger than the larger seed std on all four final metrics) is unchanged and is applied to all three seeds of each arm.
