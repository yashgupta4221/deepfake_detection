# OSDFD reproduction (CLIP ViT-L/14 + LoRA + CDC adapter + Forgery Style Mixture + Single-Center Loss)
Paper: Kong et al., "Open-Set Deepfake Detection: A Parameter-Efficient Adaptation Method With Forgery Style Mixture", IEEE TCSVT 36(7), 2026.
Files: osdfd.py (model/FSM/SCL), cache_tokens.py, train_osdfd.py, eval_osdfd.py, cdf_crops.py, extract_subset.py, test_mock.py, NOTEBOOK_CELLS.md.
Verified locally: block-by-block forward == HuggingFace CLIP (max diff 0), 2,890,689 trainable params at K=0 (paper: 2.89M), mock end-to-end run.

Deviations from the paper (be explicit when reporting):
1. LoRA+adapters only in blocks K..23 (default K=12) so the frozen prefix can be cached -> ~5x faster. K=0 is the paper's design.
2. Training stops after a time budget (default 45 min, a few thousand iterations) instead of 30,000 iterations; lr 1e-4 instead of 3e-5 for that reason.
3. Training pool: ~7,000 real + ~9,400 fake frames sampled from GenD's FF++ c23 x1.3 crops (not all frames); batch 48 = 24 real + 6 per forgery domain.
4. Unspecified in the paper, chosen here: adapter bottleneck m=28 and MLP head 1024-128-64-1 (picked so K=0 gives 2.89M), CLS token feeds the head,
   adapter output is zero for CLS, no activation inside the adapter (as in Eq. 2). The last block's adapter gets no gradient (head reads CLS only).
5. Face crops: GenD's RetinaFace-aligned x1.3 (train, DFDC) and MTCNN x1.3 (Celeb-DF) instead of dlib x1.3.
6. Reported checkpoint = last iteration (fixed in advance). A best-FF++-validation checkpoint is also saved. No test-set selection or tuning.

## Hybrid (OSDFD x MoE-FFD)  -  hybrid.py
One network: frozen CLIP ViT-L/14, blocks K..23 carry MoE-LoRA (q/k/v) + MoE conv adapter (parallel to the MLP) as in MoE-FFD
(IEEE TDSC 2026, Kong et al.); training uses OSDFD's FSM + Single-Center Loss + domain-balanced batches. FSM is applied to the
hidden states BEFORE the routed blocks (`--fsm_layers`, default K,24) so the routers are trained on style-mixed features.
Loss = BCE + lam*SCL + moe_coef*balance. Reductions: 1 expert + FSM + SCL ~ OSDFD ; many experts + `--fsm_p 0 --lam 0` ~ MoE-FFD.

    python train_osdfd.py --cache C --out O --arch hybrid                     # full hybrid (FSM at 12 and 24)
    python train_osdfd.py --cache C --out O --arch hybrid --fsm_p 0 --lam 0   # ablation: MoE experts + CE only (MoE-FFD-like)
    python train_osdfd.py --cache C --out O --arch hybrid --fsm_layers 24     # ablation: FSM not seen by routers (OSDFD placement)
    python eval_osdfd.py --ckpt O/osdfd_last.pt ...                           # reads the architecture from the checkpoint

Deviations from MoE-FFD (their code: github.com/LoveSiameseCat/MoE-FFD, read but NOT copied): straight-through gate so the task
loss trains the router (with softmax over the top-1 logit the gate is constant 1); Switch-style balance loss instead of CV^2;
4 LoRA experts (ranks 4/8/16/32) + 3 conv experts (vanilla/central-diff/angular-diff) instead of 7 + 5; CLIP ViT-L/14 with a 16x16
grid instead of ImageNet ViT-B with 14x14; experts only in blocks K..23 (frozen prefix cached). Tests: test_hybrid.py, test_e2e_mock.py.

## Round 2 (per-image routing, held-out-forgery selection, seeds)
See `PROTOCOL.md` (rules fixed before running). `round2.py --stage A|B` runs everything (resumable), `summarize.py` applies the
selection / verdict rules. New flags in `train_osdfd.py`: `--lora_route token|image`, `--holdout DF|F2F|FS|NT`, `--split_seed`.
Correction to the round-1 README: MoE-FFD is IEEE Transactions on Dependable and Secure Computing, 2026, vol. 23, no. 1 (not 2025).
