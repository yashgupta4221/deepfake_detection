"""fdnet.py - a small, from-scratch, image-only deepfake detector built to be evaluated STRICTLY cross-dataset
(train: FaceForensics++ c23 only; test: DFDC, Celeb-DF v2, never used for training or tuning).

Backbone: frozen CLIP ViT-L/14 (Radford et al. 2021; HuggingFace `openai/clip-vit-large-patch14`) with LoRA (Hu et al. 2022) on the
attention q/v projections. Everything below the LoRA/head is our own code. Three independent, switchable components (all OFF = plain
LoRA + BCE baseline), so every claim can be ablated:

  --fuse   last | multi   multi = learned softmax-weighted fusion of CLS+mean-patch features from several intermediate layers
  --grp    none | dro     dro   = worst-group training over the groups {real, DF, F2F, FS, NT} (exponentiated-gradient group weights,
                                  initialised at the plain batch proportions so step 0 is identical to the baseline)
  eval --merge none | soup | ens
                          soup = average the weights of several runs that share --init_seed (different data order),
                          ens  = average the predicted probabilities of those runs

Design choices that make results trustworthy:
  * --init_seed fixes LoRA-A and head initialisation (needed for weight averaging); --seed controls data order only.
  * No augmentation by default, constant lr, FIXED iteration budget, LAST checkpoint only (no checkpoint picking on any test data).
  * eval saves per-frame scores so any video-level partition can be recomputed later; DFDC videos are split deterministically
    (md5 of video id) into a DEV part (selection of the final configuration) and a FINAL part; Celeb-DF v2 is FINAL only.

  python fdnet.py train --manifest ffpp/manifest.csv --out runs/a_s0 --seed 0 --init_seed 0 [--fuse multi] [--grp dro]
  python fdnet.py eval  --ckpt runs/a_s0/last.pt [more ckpts] --merge none --fuse ... --sets dfdc=dfdc/manifest.csv:dev cdf2=cdf2/manifest.csv --out res.json"""
import argparse, csv, hashlib, json, math, os, random, time
from collections import defaultdict
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torch.utils.checkpoint
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import transforms

FAKES = ["DF", "F2F", "FS", "NT"]; GROUPS = ["real"] + FAKES
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
AMP = torch.float16 if DEV == "cuda" else torch.bfloat16


def autocast(): return torch.autocast(DEV, dtype=AMP)


# ----------------------------------------------------------------------------------------------------------- model
class LoRALinear(nn.Module):
    """y = W x + (alpha/r) * B A x ; W frozen, A,B float32 trainable, B=0 at init (so the model starts identical to CLIP)."""
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32)); nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32))
    def forward(self, x):
        y = self.base(x)
        return y + ((x.to(self.A.dtype) @ self.A.t()) @ self.B.t() * self.scale).to(y.dtype)


class Detector(nn.Module):
    def __init__(self, clip, r=8, alpha=16, fuse="last", layers=(8, 12, 16, 20, 24), proj=256, init_seed=0):
        super().__init__()
        self.clip = clip; cfg = clip.config; vm = getattr(clip, "vision_model", clip)
        nl = len(vm.encoder.layers); d = cfg.hidden_size
        self.layers = (nl,) if fuse == "last" else tuple(sorted(set(min(l, nl) for l in layers)))
        for p in self.clip.parameters(): p.requires_grad = False
        self.gc = False                                      # gradient checkpointing (saves memory, ~30% slower); toggled by train()
        for lay in vm.encoder.layers: self._wrap(lay)
        torch.manual_seed(init_seed)                         # shared LoRA-A / head initialisation across runs of a soup
        for lay in vm.encoder.layers:
            lay.self_attn.q_proj = LoRALinear(lay.self_attn.q_proj, r, alpha); lay.self_attn.v_proj = LoRALinear(lay.self_attn.v_proj, r, alpha)
        self.lns = nn.ModuleList([nn.LayerNorm(2 * d) for _ in self.layers])
        self.projs = nn.ModuleList([nn.Linear(2 * d, proj) for _ in self.layers])
        self.gate = nn.Parameter(torch.zeros(len(self.layers)))     # softmax -> layer weights (uniform at init)
        self.mlp = nn.Sequential(nn.Linear(proj, 128), nn.ReLU(), nn.Linear(128, 1))

    def _wrap(self, layer):
        orig = layer.forward
        def fwd(*args, **kw):
            if self.gc and self.training and torch.is_grad_enabled():
                return torch.utils.checkpoint.checkpoint(orig, *args, use_reentrant=False, **kw)
            return orig(*args, **kw)
        layer.forward = fwd

    def forward(self, pix):
        hs = self.clip(pixel_values=pix, output_hidden_states=True).hidden_states     # hs[0]=embeddings, hs[i]=output of layer i
        w = torch.softmax(self.gate, 0); f = 0
        for k, l in enumerate(self.layers):
            t = hs[l].float(); v = torch.cat([t[:, 0], t[:, 1:].mean(1)], -1)
            f = f + w[k] * F.gelu(self.projs[k](self.lns[k](v)))
        return self.mlp(f).squeeze(1)

    def trainable_state(self): return {k: v.detach().cpu() for k, v in self.state_dict().items() if ".A" in k[-3:] or k.endswith(".B") or not k.startswith("clip.")}


def load_clip(name="openai/clip-vit-large-patch14"):
    from transformers import CLIPVisionModel
    dt = torch.float16 if torch.cuda.is_available() else torch.float32
    try: return CLIPVisionModel.from_pretrained(name, dtype=dt)
    except TypeError: return CLIPVisionModel.from_pretrained(name, torch_dtype=dt)


def build(a, clip=None):
    clip = clip if clip is not None else load_clip(a.clip)
    return Detector(clip, r=a.r, alpha=a.alpha, fuse=a.fuse, init_seed=a.init_seed).to(DEV)


def count_trainable(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ------------------------------------------------------------------------------------------------------------ data
class FrameDS(torch.utils.data.Dataset):
    def __init__(self, paths, labels, groups, flip=False, size=224):
        self.p, self.y, self.g = paths, labels, groups
        tf = [transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC)] + ([transforms.RandomHorizontalFlip()] if flip else [])
        self.t = transforms.Compose(tf + [transforms.ToTensor(), transforms.Normalize(CLIP_MEAN, CLIP_STD)])
    def __len__(self): return len(self.p)
    def __getitem__(self, i): return self.t(Image.open(self.p[i]).convert("RGB")), self.y[i], self.g[i]


class InfBatches:
    """endless batches: n_real real frames + n_per_fake frames from each forgery type; each pool cycled in random order."""
    def __init__(self, pools, n_real, n_per_fake, seed):
        self.pools, self.nr, self.nf, self.rng = pools, n_real, n_per_fake, np.random.RandomState(seed); self.cyc = {s: [] for s in pools}
    def take(self, s, n):
        while len(self.cyc[s]) < n: self.cyc[s] += list(self.rng.permutation(self.pools[s]))
        r, self.cyc[s] = self.cyc[s][:n], self.cyc[s][n:]
        return [int(i) for i in r]
    def __iter__(self):
        while True: yield self.take("real", self.nr) + [i for s in FAKES for i in self.take(s, self.nf)]


def read_manifest(p): return list(csv.DictReader(open(p)))


def heldout_videos(rows, split_seed=0, val_frac=0.1):
    """the FF++ videos held out from training (same video-level shuffle that train() uses) -> an in-domain TEST set never trained on."""
    vids = sorted({r["video"] for r in rows}); random.Random(split_seed).shuffle(vids)
    return set(vids[:max(1, int(len(vids) * val_frac))])


def part_of(video):
    """deterministic video-level partition of an evaluation set: ~1/3 'dev' (used to choose the configuration), ~2/3 'final'."""
    return "dev" if int(hashlib.md5(str(video).encode()).hexdigest(), 16) % 3 == 0 else "final"


# --------------------------------------------------------------------------------------------------------- training
def group_loss(logit, y, g, q, use_dro, eta, floor=0.02):
    """per-sample BCE -> per-group means. plain: ordinary mean. dro: sum_g q_g * loss_g with q updated by exponentiated gradient."""
    l = F.binary_cross_entropy_with_logits(logit.float(), y, reduction="none")
    if not use_dro: return l.mean(), q, None
    gl = torch.stack([l[g == k].mean() for k in range(len(GROUPS))])
    loss = (q * gl).sum()
    q = q * torch.exp(eta * gl.detach()); q = torch.clamp(q / q.sum(), min=floor); q = q / q.sum()
    return loss, q, gl.detach()


def train(a, clip=None):
    os.makedirs(a.out, exist_ok=True); rows = read_manifest(a.manifest)
    val_v = heldout_videos(rows, a.split_seed, a.val_frac)
    is_val = np.array([r["video"] in val_v for r in rows]); src = np.array([r["source"] for r in rows])
    pools = {s: np.where((src == s) & ~is_val)[0] for s in GROUPS}; val_idx = np.where(is_val & np.isin(src, GROUPS))[0]
    assert all(len(v) > 0 for v in pools.values()), "empty pool - check the 'source' column"
    print({s: len(v) for s, v in pools.items()}, "| val frames", len(val_idx), flush=True)
    labels = [int(r["label"]) for r in rows]; gid = [GROUPS.index(r["source"]) if r["source"] in GROUPS else 0 for r in rows]
    ds = FrameDS([r["path"] for r in rows], labels, gid, flip=a.flip)
    vds = FrameDS([r["path"] for r in rows], labels, gid, flip=False)
    model = build(a, clip); model.gc = (a.gc == "1")
    print("trainable parameters:", f"{count_trainable(model):,}", "| fused layers:", model.layers, flush=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)                    # data order / dropout randomness only
    dl = torch.utils.data.DataLoader(ds, batch_sampler=InfBatches(pools, a.n_real, a.n_per_fake, a.seed), num_workers=a.workers, pin_memory=(DEV == "cuda"))
    vdl = torch.utils.data.DataLoader(torch.utils.data.Subset(vds, [int(i) for i in val_idx]), batch_size=96, num_workers=a.workers)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=a.lr, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler("cuda", enabled=(DEV == "cuda"))
    tot = a.n_real + 4 * a.n_per_fake; q = torch.tensor([a.n_real / tot] + [a.n_per_fake / tot] * 4, device=DEV)   # = plain-loss weights
    log, t0, run, it = [], time.time(), 0.0, 0

    @torch.no_grad()
    def val_auc():
        model.eval(); ps, ys = [], []
        for x, y, _ in vdl:
            with autocast(): ps.append(torch.sigmoid(model(x.to(DEV)).float()).cpu())
            ys.append(y)
        model.train(); return float(roc_auc_score(torch.cat(ys).numpy(), torch.cat(ps).numpy()))

    model.train(); data = iter(dl)
    for it in range(1, a.iters + 1):
        x, y, g = next(data); x, y, g = x.to(DEV), y.float().to(DEV), g.to(DEV)
        for attempt in (0, 1):
            try:
                with autocast(): logit = model(x)
                loss, q_new, gl = group_loss(logit, y, g, q, a.grp == "dro", a.eta)
                opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); break
            except torch.cuda.OutOfMemoryError:
                if it != 1 or a.gc != "auto" or model.gc or attempt: raise
                print("out of memory -> switching on gradient checkpointing and retrying this batch", flush=True)
                opt.zero_grad(set_to_none=True); logit = loss = None; torch.cuda.empty_cache(); model.gc = True
        q = q_new; scaler.step(opt); scaler.update(); run += float(loss.detach())
        if it % 50 == 0 or it == a.iters:
            extra = f" | q {[round(float(v), 3) for v in q]}" if a.grp == "dro" else ""
            print(f"it {it} | loss {run / 50:.4f} | {(time.time() - t0) / 60:.1f} min | {it / (time.time() - t0):.2f} it/s{extra}", flush=True); run = 0.0
        if it % a.eval_every == 0 or it == a.iters:
            v = val_auc(); log.append({"it": it, "val_auc": v}); print(f"   FF++ val AUC {v:.4f}", flush=True)
            torch.save({"model": model.trainable_state(), "it": it, "args": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in vars(a).items()}}, f"{a.out}/last.pt")
            json.dump(log, open(f"{a.out}/train_log.json", "w"))
    json.dump({"iters": it, "last_val_auc": log[-1]["val_auc"], "args": {k: str(v) for k, v in vars(a).items()}}, open(f"{a.out}/DONE.json", "w"))
    print("finished", it, "iterations", flush=True)


# ---------------------------------------------------------------------------------------------------------- evaluation
def merge_states(states):
    keys = states[0].keys(); assert all(s.keys() == keys for s in states), "checkpoints have different parameters"
    return {k: sum(s[k].float() for s in states) / len(states) for k in keys}


@torch.no_grad()
def score_set(model, rows, bs, workers):
    dl = torch.utils.data.DataLoader(FrameDS([r["path"] for r in rows], [int(r["label"]) for r in rows], [0] * len(rows)), batch_size=bs, num_workers=workers)
    model.eval(); ps = []
    for i, (x, _, _) in enumerate(dl):
        with autocast(): ps.append(torch.sigmoid(model(x.to(DEV)).float()).cpu())
        if i % 50 == 0: print(i * bs, "/", len(rows), flush=True)
    return torch.cat(ps).numpy()


def metrics(rows, p):
    y = np.array([int(r["label"]) for r in rows]); vp = defaultdict(list); vy = {}
    for pi, yi, r in zip(p, y, rows): vp[r["video"]].append(pi); vy[r["video"]] = yi
    ks = list(vp)
    return {"n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum()), "n_videos": len(ks), "frame_auroc": float(roc_auc_score(y, p)),
            "frame_acc@0.5": float(((p > 0.5) == y).mean()), "video_auroc": float(roc_auc_score([vy[k] for k in ks], [np.mean(vp[k]) for k in ks]))}


def evaluate(a, clip=None):
    clip = clip if clip is not None else load_clip(a.clip)
    cks = [torch.load(c, map_location="cpu")["model"] for c in a.ckpt]
    if a.merge == "none": assert len(cks) == 1, "--merge none needs exactly one checkpoint"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    def make(state):
        m = build(a, clip); r = m.load_state_dict(state, strict=False)
        assert not r.unexpected_keys and all(k.startswith("clip.") for k in r.missing_keys), (r.unexpected_keys, [k for k in r.missing_keys if not k.startswith("clip.")])
        return m
    model = make(merge_states(cks) if a.merge == "soup" else cks[0])      # built ONCE: LoRA wrapping mutates the CLIP object in place
    res = {}
    for spec in a.sets:
        name, rest = spec.split("=", 1); man, _, part = rest.partition(":")
        rows = read_manifest(man)
        if part == "ffval":      # in-domain FF++ test = the videos held out from training (not the official FF++ test split)
            hv = heldout_videos(rows, a.split_seed, a.val_frac); rows = [r for r in rows if r["video"] in hv and r["source"] in GROUPS]
        elif part: rows = [r for r in rows if part_of(r["video"]) == part]
        if a.merge == "ens":
            ps = []
            for c in cks:
                r_ = model.load_state_dict(c, strict=False); assert not r_.unexpected_keys
                ps.append(score_set(model, rows, a.bs, a.workers))
            p = np.mean(ps, 0)
        else: p = score_set(model, rows, a.bs, a.workers)
        res[name] = metrics(rows, p); print(name, res[name], flush=True)
        if a.save_scores:
            os.makedirs(a.save_scores, exist_ok=True)
            with open(os.path.join(a.save_scores, f"{name}.csv"), "w", newline="") as f:
                w = csv.writer(f); w.writerow(["path", "label", "video", "score"])
                for r, s in zip(rows, p): w.writerow([r["path"], r["label"], r["video"], f"{float(s):.6f}"])
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["train", "eval"])
    ap.add_argument("--clip", default="openai/clip-vit-large-patch14"); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fuse", default="last", choices=["last", "multi"]); ap.add_argument("--grp", default="none", choices=["none", "dro"])
    ap.add_argument("--r", type=int, default=8); ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--manifest", default=""); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--init_seed", type=int, default=0)
    ap.add_argument("--split_seed", type=int, default=0); ap.add_argument("--val_frac", type=float, default=0.1)   # must match between train and eval
    ap.add_argument("--iters", type=int, default=2000); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--eta", type=float, default=0.01)
    ap.add_argument("--n_real", type=int, default=24); ap.add_argument("--n_per_fake", type=int, default=6); ap.add_argument("--flip", action="store_true")
    ap.add_argument("--eval_every", type=int, default=250); ap.add_argument("--gc", default="auto", choices=["auto", "0", "1"])
    ap.add_argument("--ckpt", nargs="*", default=[]); ap.add_argument("--merge", default="none", choices=["none", "soup", "ens"])
    ap.add_argument("--sets", nargs="*", default=[]); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--save_scores", default="")
    a = ap.parse_args()
    train(a) if a.mode == "train" else evaluate(a)
