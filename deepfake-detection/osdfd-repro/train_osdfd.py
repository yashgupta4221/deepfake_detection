"""Train OSDFD on cached tokens. Paper recipe: Adam (b1 .9, b2 .999), CONSTANT lr, batch 48, lambda(SCL)=1, FSM active with
probability 0.5, batch = real faces + equal numbers of faces from each of the 4 forgery domains (real:fake = 1:1, i.e.
the paper's 'real faces augmented four-fold'). Training stops after --minutes (time budget) or --max_iters.
Checkpoints: osdfd_last.pt (primary, decided in advance) and osdfd_best_val.pt (best FF++ validation AUC, like the paper)."""
import argparse, json, os, queue, threading, time, numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from osdfd import OSDFD, single_center_loss, count_trainable, DEV, autocast
from hybrid import HybridOSDFD, build_model

FAKES = ["DF", "F2F", "FS", "NT"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--clip_name", default="openai/clip-vit-large-patch14")
    ap.add_argument("--minutes", type=float, default=45); ap.add_argument("--max_iters", type=int, default=30000)
    ap.add_argument("--lr", type=float, default=1e-4, help="paper: 3e-5 for 30k its; raised for the ~10x shorter schedule")
    ap.add_argument("--n_real", type=int, default=24); ap.add_argument("--n_per_fake", type=int, default=6)
    ap.add_argument("--lam", type=float, default=1.0); ap.add_argument("--fsm_p", type=float, default=0.5)
    ap.add_argument("--val_frac", type=float, default=0.1); ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--resume", default="")
    ap.add_argument("--arch", choices=["osdfd", "hybrid"], default="osdfd")
    ap.add_argument("--lora_ranks", default="4,8,16,32", help="hybrid: LoRA expert ranks")
    ap.add_argument("--adp_experts", default="cv,cd,ad", help="hybrid: conv expert types (cv vanilla, cd central-diff, ad angular-diff)")
    ap.add_argument("--fsm_layers", default="", help="hybrid: block indices whose INPUT gets FSM (24 = after last block). default: K,24")
    ap.add_argument("--moe_coef", type=float, default=0.01, help="hybrid: weight of the router balance loss")
    ap.add_argument("--lora_route", choices=["token", "image"], default="token", help="hybrid: LoRA router granularity (round 2: image)")
    ap.add_argument("--holdout", default="", choices=["", "DF", "F2F", "FS", "NT"],
                    help="exclude this forgery type from TRAINING; its held-out AUROC (vs real validation frames) becomes the selection metric")
    ap.add_argument("--split_seed", type=int, default=0, help="seed of the train/val VIDEO split (kept fixed across --seed runs)")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rng = np.random.RandomState(a.seed); torch.manual_seed(a.seed)
    FK = [f for f in FAKES if f != a.holdout]                                   # forgery types used for training
    if a.holdout and a.n_per_fake == 6:                                         # keep real:fake = 1:1 and batch size 48
        a.n_per_fake = a.n_real // len(FK)

    meta = json.load(open(a.cache + "_meta.json")); K = meta["K"]
    tok = np.load(a.cache + "_tok.npy", mmap_mode="r")
    lab = np.load(a.cache + "_labels.npy"); src = np.load(a.cache + "_sources.npy"); vid = np.load(a.cache + "_videos.npy")
    # split by VIDEO into train / val so FF++ validation has no frame leakage
    uv = np.unique(vid); np.random.RandomState(a.split_seed).shuffle(uv); val_v = set(uv[: int(len(uv) * a.val_frac)])
    is_val = np.array([v in val_v for v in vid])
    pools = {s: np.where((src == s) & ~is_val)[0] for s in ["real"] + FK}
    val_idx = np.where(is_val)[0]
    if a.holdout:                                                               # validation frames: real + trained forgeries only
        val_idx = val_idx[src[val_idx] != a.holdout]
        ho_idx = np.where(is_val & ((src == a.holdout) | (src == "real")))[0]   # held-out forgery vs real, never trained on
        print("held-out forgery:", a.holdout, "| frames", int((src[ho_idx] == a.holdout).sum()), flush=True)
    print({s: len(v) for s, v in pools.items()}, "| val frames", len(val_idx), "| K =", K, flush=True)
    json.dump({"pool_sizes": {s: int(len(v)) for s, v in pools.items()}, "val_frames": int(len(val_idx)), "K": K},
              open(f"{a.out}/data_split.json", "w"))

    cfg = None
    if a.arch == "hybrid":
        cfg = dict(arch="hybrid", lora_ranks=[int(x) for x in a.lora_ranks.split(",")], adapter_experts=a.adp_experts.split(","), m=28, lora_route=a.lora_route,
                   fsm_layers=[int(x) for x in a.fsm_layers.split(",")] if a.fsm_layers else [K, 24])
    model = build_model(cfg, a.clip_name, K).to(DEV)
    print("config:", cfg or "osdfd", flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    print("trainable parameters:", f"{count_trainable(model):,}", flush=True)
    opt = torch.optim.Adam(params, lr=a.lr, betas=(0.9, 0.999))              # constant lr, no decay (paper)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEV == "cuda"))
    it0 = 0
    if a.resume:
        ck = torch.load(a.resume, map_location=DEV); model.load_state_dict(ck["model"], strict=False)
        opt.load_state_dict(ck["opt"]); it0 = ck["it"]; print("resumed at", it0)

    # ---- batch sampler with a prefetch thread ----
    cyc = {s: [] for s in pools}
    def take(s, n):
        while len(cyc[s]) < n: cyc[s] += list(rng.permutation(pools[s]))
        r, cyc[s] = cyc[s][:n], cyc[s][n:]
        return r
    def make_batch():
        idx, y, dom = [], [], []
        idx += take("real", a.n_real); y += [0] * a.n_real; dom += [-1] * a.n_real
        for d, s in enumerate(FK):
            idx += take(s, a.n_per_fake); y += [1] * a.n_per_fake; dom += [d] * a.n_per_fake
        order = np.argsort(idx); idx = np.array(idx)[order]
        x = torch.from_numpy(np.ascontiguousarray(tok[idx]))
        x = x.pin_memory() if DEV == "cuda" else x
        return x, torch.tensor(np.array(y)[order]).float(), torch.tensor(np.array(dom)[order])
    q = queue.Queue(maxsize=6)
    def producer():
        while True: q.put(make_batch())
    threading.Thread(target=producer, daemon=True).start()

    @torch.no_grad()
    def val_auc():
        model.eval(); ps = []
        for i in range(0, len(val_idx), 128):
            x = torch.from_numpy(np.ascontiguousarray(tok[val_idx[i:i + 128]])).to(DEV)
            with autocast():
                ps.append(torch.sigmoid(model.forward_tokens(x)[0]).float().cpu())
        model.train()
        return float(roc_auc_score(lab[val_idx], torch.cat(ps).numpy()))

    @torch.no_grad()
    def holdout_auc():
        model.eval(); ps = []
        for i in range(0, len(ho_idx), 128):
            x = torch.from_numpy(np.ascontiguousarray(tok[ho_idx[i:i + 128]])).to(DEV)
            with autocast():
                ps.append(torch.sigmoid(model.forward_tokens(x)[0]).float().cpu())
        model.train()
        return float(roc_auc_score(lab[ho_idx], torch.cat(ps).numpy()))

    log, best, t0, model_train = [], -1, time.time(), model.train()
    run_bce = run_scl = run_aux = 0.0
    for it in range(it0 + 1, a.max_iters + 1):
        x, y, dom = q.get(); x, y, dom = x.to(DEV), y.to(DEV), dom.to(DEV)
        use_fsm = bool(rng.rand() < a.fsm_p)
        with autocast():
            logit, z = model.forward_tokens(x, dom, y == 1, use_fsm=use_fsm)
        bce = F.binary_cross_entropy_with_logits(logit.float(), y); scl = single_center_loss(z.float(), y.long())
        loss = bce + a.lam * scl
        if a.arch == "hybrid":
            aux = model.aux_loss(); loss = loss + a.moe_coef * aux; run_aux += float(aux)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        run_bce += float(bce); run_scl += float(scl)
        el = time.time() - t0
        done = el > a.minutes * 60 or it == a.max_iters
        if it % 50 == 0 or done:
            extra = ""
            if a.arch == "hybrid":
                u = model.usage(); la = np.array(u["lora"]).mean(0).round(2).tolist(); ad = np.array(u["adapter"]).mean(0).round(2).tolist()
                extra = f" bal {run_aux / 50:.3f} | expert use lora {la} adp {ad}"
            print(f"it {it} | bce {run_bce / 50:.4f} scl {run_scl / 50:.4f}{extra} | {el / 60:.1f} min | {(it - it0) / el:.2f} it/s", flush=True)
            log.append({"it": it, "bce": run_bce / 50, "scl": run_scl / 50, "min": el / 60}); run_bce = run_scl = run_aux = 0.0
        if it % a.eval_every == 0 or done:
            v = val_auc(); rec = {"it": it, "val_auc": v}; msg = f"   FF++ val AUC {v:.4f}"
            if a.holdout:
                h = holdout_auc(); rec["holdout_auc"] = h; msg += f" | held-out {a.holdout} AUC {h:.4f}"
            log.append(rec); print(msg, flush=True)
            if v > best:
                best = v; torch.save({"model": model.trainable_state(), "K": K, "it": it, "cfg": cfg}, f"{a.out}/osdfd_best_val.pt")
            torch.save({"model": model.trainable_state(), "opt": opt.state_dict(), "K": K, "it": it, "cfg": cfg}, f"{a.out}/osdfd_last.pt")
            json.dump(log, open(f"{a.out}/train_log.json", "w"))
        if done: break
    print("finished", it, "iterations; best val AUC", best)
    json.dump({"iters": it, "args": vars(a), "last_val_auc": log[-1].get("val_auc") if log else None,
               "last_holdout_auc": next((r["holdout_auc"] for r in reversed(log) if "holdout_auc" in r), None)}, open(f"{a.out}/DONE.json", "w"))


if __name__ == "__main__":
    main()
