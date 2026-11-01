# Round-2 protocol (written BEFORE any round-2 run; do not edit after seeing results)

## Why round 2 exists
Round 1 (one seed, one run per variant, ~2,340 its for hybrids vs 3,432 for OSDFD) gave:
OSDFD DFDC 0.805-0.812 / 0.823-0.834 (frame/video AUROC), Celeb-DF 0.870 / 0.931; hybrid variants DFDC 0.805-0.817 / 0.834-0.842,
Celeb-DF 0.810-0.843 / 0.858-0.900. No hybrid beat OSDFD on Celeb-DF. Round 2 tests whether fixing the known weaknesses of round 1 changes that.

## Data hygiene (the important part)
* DFDC and Celeb-DF test numbers for the round-1 variants have been SEEN. Round 2 therefore selects the hybrid variant **without
  touching any test set**: Stage A uses FF++ only, with one forgery type (NeuralTextures, NT) held out of training and used as the
  generalisation proxy (held-out NT frames vs real validation frames, split by video).
* Test sets are evaluated only in Stage B, once per run, using the LAST checkpoint (no best-checkpoint picking).
* Every variant tried is reported, including failures. Round-1 numbers stay in the report as round 1.

## Changes tested (decided in advance)
1. Per-image LoRA routing (`--lora_route image`): one expert per image from the mean patch token, like the adapter router and like
   MoE-FFD's image-level routing. Round 1 routed per token.
2. Iteration matching: every arm in a stage trains the same number of iterations (A: 2000, B: 3432, = OSDFD's round-1 budget).
3. Seeds: Stage B uses seeds 0,1,2 for both OSDFD and the chosen hybrid. The FF++ train/val video split is fixed (`--split_seed 0`).

## Stage A - selection (FF++ only, NT held out, 2000 its, seed 0, lr 1e-4, batch 24 real + 8 per trained forgery)
Arms: `osdfd` (reference only), `hyb_token` (= round-1 hyb_full), `hyb_image`, `hyb_token_moe` (= round-1 hyb_moe_only, no FSM/SCL),
`hyb_image_moe`.
Selection metric: held-out NT AUROC at the LAST checkpoint. Pick the hybrid arm with the highest value; if several are within 0.005,
take the first in this order: hyb_image, hyb_token, hyb_image_moe, hyb_token_moe (keep FSM+SCL, since that is what makes it a hybrid).
Stop rule: if OSDFD's held-out AUROC exceeds the best hybrid's by more than 0.02, the hybrid is not expected to beat OSDFD; Stage B is
then optional and the Stage A table is itself the finding.

## Stage B - final (full FF++, 2000 its, seeds 0,1,2; OSDFD and the selected hybrid)
Metrics: frame and video AUROC on DFDC (40,000 frames, 4,996 videos) and Celeb-DF v2 (6,869 frames, 518 videos). Report mean +- sample std
over the 3 seeds and every per-seed value.
Verdict (hard-coded in summarize.py): the hybrid counts as better (worse) than OSDFD only if its mean differs from OSDFD's by more
than the larger of the two seed-stds on ALL FOUR metrics. Anything else is reported as "no reliable difference".

## Known limits (state them in any write-up)
* n = 3 seeds: std estimates are crude. Only two test sets. Celeb-DF has 518 videos (noisy).
* The proxy (one held-out forgery type) may not rank variants the way DFDC/Celeb-DF do.
* ~1.5M (OSDFD) vs ~5.6M (hybrid) trainable parameters: capacity differs; this is part of the comparison, not hidden.
* Lower learning-rate / longer schedules (paper: 30k its, lr 3e-5) were not run; all results are on a ~10x shorter schedule.

## Deviation recorded BEFORE Stage B was run (2026-10-18)
* Stage B uses `--iters_b 2000` instead of 3432 to fit the GPU budget (about 5 h instead of 7.5 h). Both arms use the same 2000
  iterations (= the Stage A budget). Seeds (3), data, metrics, last-checkpoint rule and verdict rule are unchanged.
* Consequence to state in any write-up: all Stage B conclusions are for this ~15x shorter schedule (paper: 30k its); OSDFD numbers are
  not comparable to its round-1 numbers (3432 its).
* Stage A result: selected arm `hyb_token` (held-out NT AUC 0.6705; OSDFD 0.5859; hyb_image 0.6227; hyb_token_moe 0.5891; hyb_image_moe 0.5272).
