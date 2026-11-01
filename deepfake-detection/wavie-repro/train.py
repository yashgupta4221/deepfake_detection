"""Train WAVIE on cached CLIP [CLS] features. Paper protocol (Sec. IV-C): 20 epochs, AdamW lr 1e-3 wd 1e-2,
batch 32, AMP, OneCycleLR, EMA 0.999, loss = BCE + 0.1*SupCon(tau 0.07), no augmentation, fixed seeds,
80/20 random frame split with a fixed seed. Final-epoch EMA weights are used (no test-set selection).
Usage: python train.py --feats feats/ffpp --out runs/wavie [--mode full] [--epochs 20]"""
import argparse, json, os, random, copy, numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from model import WAVIE, sup_con


def load(prefix, in_ram):
    y = np.load(prefix + "_labels.npy")
    X = np.load(prefix + "_cls.npy", mmap_mode="r")
    return (np.ascontiguousarray(X) if in_ram else X), y


@torch.no_grad()
def predict(model, X, dev, bs=1024):
    model.eval(); out = []
    for i in range(0, len(X), bs):
        g = torch.from_numpy(np.asarray(X[i:i + bs])).to(dev)
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            out.append(torch.sigmoid(model(g)[0]).float().cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="full"); ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2); ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--ema", type=float, default=0.999); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--supcon_on", default="q2", choices=["q2", "idwt"])  # paper text vs equation differ
    ap.add_argument("--split", default="frame", choices=["frame", "video"])  # paper: random frame split (leaky); video = no leakage
    ap.add_argument("--in_ram", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.deterministic = True
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = load(a.feats, a.in_ram)
    rng = np.random.RandomState(a.seed)
    if a.split == "frame":
        perm = rng.permutation(len(y)); ntr = int(0.8 * len(y)); tr, va = np.sort(perm[:ntr]), np.sort(perm[ntr:])
    else:
        vids = np.load(a.feats + "_videos.npy"); uv = rng.permutation(np.unique(vids)); hold = set(uv[: int(0.2 * len(uv))])
        isv = np.array([v in hold for v in vids]); tr, va = np.where(~isv)[0], np.where(isv)[0]
    json.dump({"train": int(len(tr)), "val": int(len(va))}, open(f"{a.out}/split.json", "w"))
    print(f"train {len(tr)} (real {(y[tr]==0).sum()}, fake {(y[tr]==1).sum()}) | val {len(va)}")

    model = WAVIE(mode=a.mode).to(dev)
    ntrain = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {ntrain:,}")
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * (len(tr) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    log = []
    for ep in range(a.epochs):
        model.train(); order = np.random.RandomState(a.seed + ep).permutation(tr)
        tot = 0.0
        for i in range(len(tr) // a.bs):
            idx = np.sort(order[i * a.bs:(i + 1) * a.bs])
            g = torch.from_numpy(np.asarray(X[idx])).to(dev); t = torch.from_numpy(y[idx]).float().to(dev)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                logit, z, fhat = model(g)
                rep = z if a.supcon_on == "q2" else fhat
                loss = F.binary_cross_entropy_with_logits(logit.float(), t) + a.lam * sup_con(rep, t.long())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()): pe.mul_(a.ema).add_(pm.detach(), alpha=1 - a.ema)
                for be, bm in zip(ema.buffers(), model.buffers()): be.copy_(bm)
            tot += loss.item()
        pv = predict(ema, X[va], dev)
        auc = roc_auc_score(y[va], pv)
        log.append({"epoch": ep + 1, "train_loss": tot / (len(tr) // a.bs), "val_auroc_ema": float(auc)})
        print(log[-1], flush=True)
        torch.save(ema.state_dict(), f"{a.out}/wavie_ema_last.pt")   # final-epoch EMA = the reported model
    json.dump({"args": vars(a), "trainable_params": ntrain, "log": log}, open(f"{a.out}/train_log.json", "w"), indent=1)


if __name__ == "__main__":
    main()
