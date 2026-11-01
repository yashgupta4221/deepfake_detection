"""OSDFD components on an FS-VFM ViT backbone ("OSDFD-FSVFM").

What is reused from OSDFD (Kong et al., IEEE TCSVT 2026), imported unchanged from osdfd.py (already tested in round 1/2):
  * LoRA (r=8) on q/k/v, CDC adapter in parallel with the MLP (Eq. 1-5), Forgery Style Mixture (Eq. 6-9), Single-Center Loss (Eq. 10-13),
  * training recipe: Adam, constant lr, batch = 24 real + 6 per forgery domain (DF, F2F, FS, NT), FSM active with prob 0.5, lambda(SCL) = 1.
What changes: the frozen backbone is the FS-VFM ViT-B/16 (Wang et al., face-security foundation model, pretrained on VGGFace2) instead of CLIP ViT-L/14.
Only LoRA + adapters + fc_norm + head are trained; every backbone weight is frozen. No data augmentation (as in OSDFD).

  train: python fsvfm_osdfd.py train --code CODE --repo FSFM/fsvfm --pretrained .../checkpoint-600.pth --stats .../pretrain_ds_mean_std.txt \
             --manifest ffpp/manifest.csv --out runs/h_s0 --seed 0 --iters 2000
  eval : python fsvfm_osdfd.py eval  --code CODE --repo FSFM/fsvfm --pretrained ... --stats ... --ckpt runs/h_s0/last.pt \
             --sets dfdc=dfdc/manifest.csv cdf2=cdf2/manifest.csv --out results.json
Fake probability = sigmoid(logit); metrics identical to eval_osdfd.py / fsvfm_eval.py (frame AUROC, acc@0.5, video AUROC = mean frame score)."""
import argparse, csv, json, os, random, sys, threading, time
from collections import defaultdict
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import transforms

FAKES = ["DF", "F2F", "FS", "NT"]
VGG_MEAN, VGG_STD = [0.5482207536697388, 0.42340534925460815, 0.3654651641845703], [0.2789176106452942, 0.2438540756702423, 0.23493893444538116]


def load_stats(path):
    if path and os.path.exists(path):
        d = json.loads(open(path).readline()); return d["mean"], d["std"]
    print("WARNING: no stats file, using VGGFace2 defaults"); return VGG_MEAN, VGG_STD


class FrameDS(torch.utils.data.Dataset):          # eval-style transform: bicubic resize to 224x224 + normalize (no augmentation)
    def __init__(self, paths, labels, doms, mean, std, size=224):
        self.p, self.y, self.d = paths, labels, doms
        self.t = transforms.Compose([transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
                                     transforms.ToTensor(), transforms.Normalize(mean, std)])
    def __len__(self): return len(self.p)
    def __getitem__(self, i): return self.t(Image.open(self.p[i]).convert("RGB")), self.y[i], self.d[i]


class InfBatches:                                  # endless batches: n_real real + n_per_fake from each forgery domain
    def __init__(self, pools, fk, n_real, n_per_fake, seed):
        self.pools, self.fk, self.nr, self.nf, self.rng = pools, fk, n_real, n_per_fake, np.random.RandomState(seed); self.cyc = {s: [] for s in pools}
    def take(self, s, n):
        while len(self.cyc[s]) < n: self.cyc[s] += list(self.rng.permutation(self.pools[s]))
        r, self.cyc[s] = self.cyc[s][:n], self.cyc[s][n:]
        return [int(i) for i in r]
    def __iter__(self):
        while True:
            yield self.take("real", self.nr) + [i for s in self.fk for i in self.take(s, self.nf)]


def build(imp, model_name, pretrained, r=8, m=28):
    """FS-VFM ViT (frozen) + OSDFD LoRA/CDC-adapters + MLP head."""
    import models_vit
    LoRA, Adapter = imp["LoRA"], imp["Adapter"]
    vit = models_vit.__dict__[model_name](num_classes=0, drop_path_rate=0.0, global_pool=True)
    if pretrained:
        sd = torch.load(pretrained, map_location="cpu")["model"]
        msg = vit.load_state_dict(sd, strict=False)
        print("backbone load - missing:", msg.missing_keys, "| unexpected (decoder etc., fine):", len(msg.unexpected_keys), flush=True)
        assert set(msg.missing_keys) <= {"fc_norm.weight", "fc_norm.bias"}, "backbone did not load as expected"
    else:
        print("WARNING: random backbone (mock test only)", flush=True)

    class HBlock(nn.Module):
        def __init__(self, blk, grid):
            super().__init__()
            self.n1, self.attn, self.n2, self.mlp = blk.norm1, blk.attn, blk.norm2, blk.mlp
            d = self.attn.qkv.in_features; self.h = self.attn.num_heads
            self.lq, self.lk, self.lv = LoRA(d, r), LoRA(d, r), LoRA(d, r); self.adapter = Adapter(d, m, grid)
        def forward(self, x):
            B, N, d = x.shape
            h = self.n1(x)
            q, k, v = self.attn.qkv(h).chunk(3, -1)                  # timm layout: [q | k | v] along the last dim
            q, k, v = q + self.lq(h), k + self.lk(h), v + self.lv(h)
            sp = lambda t: t.reshape(B, N, self.h, d // self.h).transpose(1, 2)
            o = F.scaled_dot_product_attention(sp(q), sp(k), sp(v)).transpose(1, 2).reshape(B, N, d)
            x = x + self.attn.proj(o)
            h2 = self.n2(x)
            return x + self.mlp(h2) + self.adapter(h2)               # h_out = MLP(h) + Adapter(h)   (Eq. 1)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed, self.cls_token, self.pos_embed, self.fc_norm = vit.patch_embed, vit.cls_token, vit.pos_embed, vit.fc_norm
            d = vit.embed_dim; grid = vit.patch_embed.img_size[0] // vit.patch_embed.patch_size[0]
            self.blocks = nn.ModuleList([HBlock(b, grid) for b in vit.blocks])
            for p in self.parameters(): p.requires_grad = False        # freeze the whole FS-VFM backbone
            for b in self.blocks:
                for p in list(b.lq.parameters()) + list(b.lk.parameters()) + list(b.lv.parameters()) + list(b.adapter.parameters()):
                    p.requires_grad = True
            for p in self.fc_norm.parameters(): p.requires_grad = True
            self.head = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 64)); self.cls = nn.Linear(64, 1)
        def tokens(self, pix):
            x = self.patch_embed(pix); x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], 1) + self.pos_embed
            for b in self.blocks: x = b(x)
            return x
        def head_forward(self, tok):
            z = self.head(self.fc_norm(tok[:, 1:].mean(1)).float())  # global average pool of patch tokens (as FS-VFM's fine-tuning head)
            return self.cls(z).squeeze(1), z
        def forward(self, pix, dom=None, is_fake=None, use_fsm=False):
            t = self.tokens(pix)
            if use_fsm: t = imp["fsm"](t, dom, is_fake)
            return self.head_forward(t)
        def trainable_state(self):
            return {k: v for k, v in self.state_dict().items() if any(s in k for s in (".lq.", ".lk.", ".lv.", ".adapter.", "fc_norm.", "head.", "cls."))}
    return Net()


def get_imports(code, repo):
    sys.path.insert(0, code); sys.path.insert(0, repo)
    from osdfd import LoRA, Adapter, OSDFD, single_center_loss, count_trainable, DEV, autocast
    return dict(LoRA=LoRA, Adapter=Adapter, fsm=OSDFD.fsm, scl=single_center_loss, count=count_trainable, DEV=DEV, autocast=autocast)


def read_manifest(path):
    return list(csv.DictReader(open(path)))


def train(a):
    imp = get_imports(a.code, a.repo); DEV, autocast = imp["DEV"], imp["autocast"]
    os.makedirs(a.out, exist_ok=True); torch.manual_seed(a.seed); rng = np.random.RandomState(a.seed)
    rows = read_manifest(a.manifest); mean, std = load_stats(a.stats)
    vids = sorted({r["video"] for r in rows}); random.Random(a.split_seed).shuffle(vids)          # same video-level split as fsvfm_splits.py
    val_v = set(vids[:max(1, int(len(vids) * a.val_frac))])
    is_val = np.array([r["video"] in val_v for r in rows]); src = np.array([r["source"] for r in rows])
    pools = {s: np.where((src == s) & ~is_val)[0] for s in ["real"] + FAKES}; val_idx = np.where(is_val & np.isin(src, ["real"] + FAKES))[0]
    print({s: len(v) for s, v in pools.items()}, "| val frames", len(val_idx), flush=True)
    assert all(len(v) > 0 for v in pools.values()), "a forgery pool is empty - check the 'source' column of the manifest"
    dom_of = {s: i for i, s in enumerate(FAKES)}
    labels = [int(r["label"]) for r in rows]; doms = [dom_of.get(r["source"], -1) for r in rows]
    ds = FrameDS([r["path"] for r in rows], labels, doms, mean, std)
    dl = torch.utils.data.DataLoader(ds, batch_sampler=InfBatches(pools, FAKES, a.n_real, a.n_per_fake, a.seed), num_workers=a.workers, pin_memory=(DEV == "cuda"))
    model = build(imp, a.model, a.pretrained).to(DEV)
    print("trainable parameters:", f"{imp['count'](model):,}", flush=True)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=a.lr, betas=(0.9, 0.999))      # constant lr (paper)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEV == "cuda"))
    vdl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, [int(i) for i in val_idx]), batch_size=96, num_workers=a.workers)

    @torch.no_grad()
    def val_auc():
        model.eval(); ps, ys = [], []
        for x, y, _ in vdl:
            with autocast(): ps.append(torch.sigmoid(model(x.to(DEV))[0]).float().cpu())
            ys.append(y)
        model.train(); return float(roc_auc_score(torch.cat(ys).numpy(), torch.cat(ps).numpy()))

    log, best, t0, run_b, run_s, it = [], -1, time.time(), 0.0, 0.0, 0
    model.train(); data = iter(dl)
    for it in range(1, a.iters + 1):
        x, y, dom = next(data); x, y, dom = x.to(DEV), y.float().to(DEV), dom.to(DEV)
        use_fsm = bool(rng.rand() < a.fsm_p)
        with autocast(): logit, z = model(x, dom, y == 1, use_fsm=use_fsm)
        bce = F.binary_cross_entropy_with_logits(logit.float(), y); scl = imp["scl"](z.float(), y.long()); loss = bce + a.lam * scl
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        run_b += float(bce.detach()); run_s += float(scl.detach())
        if it % 50 == 0 or it == a.iters:
            el = time.time() - t0
            print(f"it {it} | bce {run_b / 50:.4f} scl {run_s / 50:.4f} | {el / 60:.1f} min | {it / el:.2f} it/s", flush=True); run_b = run_s = 0.0
        if it % a.eval_every == 0 or it == a.iters:
            v = val_auc(); log.append({"it": it, "val_auc": v}); print(f"   FF++ val AUC {v:.4f}", flush=True); best = max(best, v)
            torch.save({"model": model.trainable_state(), "it": it, "args": vars(a)}, f"{a.out}/last.pt"); json.dump(log, open(f"{a.out}/train_log.json", "w"))
    print("finished", it, "iterations; best val AUC", best, flush=True)
    json.dump({"iters": it, "last_val_auc": log[-1]["val_auc"] if log else None, "args": vars(a)}, open(f"{a.out}/DONE.json", "w"))


def evaluate(a):
    imp = get_imports(a.code, a.repo); DEV, autocast = imp["DEV"], imp["autocast"]
    model = build(imp, a.model, a.pretrained).to(DEV)
    if a.ckpt:
        r = model.load_state_dict(torch.load(a.ckpt, map_location="cpu")["model"], strict=False)
        bad = [k for k in r.unexpected_keys]; print("unexpected keys (should be empty):", bad, flush=True)
        assert not bad
    model.eval(); mean, std = load_stats(a.stats); res = {}
    for spec in a.sets:
        name, man = spec.split("="); rows = read_manifest(man)
        y = np.array([int(r["label"]) for r in rows]); v = np.array([r["video"] for r in rows])
        dl = torch.utils.data.DataLoader(FrameDS([r["path"] for r in rows], [int(r["label"]) for r in rows], [-1] * len(rows), mean, std), batch_size=a.bs, num_workers=a.workers)
        ps = []
        with torch.no_grad():
            for i, (x, _, _) in enumerate(dl):
                with autocast(): ps.append(torch.sigmoid(model(x.to(DEV))[0]).float().cpu())
                if i % 50 == 0: print(name, i * a.bs, "/", len(rows), flush=True)
        p = torch.cat(ps).numpy(); vp = defaultdict(list); vy = {}
        for pi, yi, vi in zip(p, y, v): vp[vi].append(pi); vy[vi] = yi
        keys = list(vp)
        res[name] = {"n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum()), "frame_auroc": float(roc_auc_score(y, p)), "frame_acc@0.5": float(((p > 0.5) == y).mean()),
                     "video_auroc": float(roc_auc_score([vy[k] for k in keys], [np.mean(vp[k]) for k in keys])), "n_videos": len(keys)}
        print(name, res[name], flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["train", "eval"])
    ap.add_argument("--code", required=True, help="folder with osdfd.py"); ap.add_argument("--repo", required=True, help="FSFM/fsvfm folder (models_vit.py)")
    ap.add_argument("--model", default="vit_base_patch16"); ap.add_argument("--pretrained", default=""); ap.add_argument("--stats", default="")
    ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--manifest", default=""); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=2000); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n_real", type=int, default=24); ap.add_argument("--n_per_fake", type=int, default=6)
    ap.add_argument("--lam", type=float, default=1.0); ap.add_argument("--fsm_p", type=float, default=0.5)
    ap.add_argument("--val_frac", type=float, default=0.1); ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--ckpt", default=""); ap.add_argument("--sets", nargs="*", default=[]); ap.add_argument("--bs", type=int, default=96)
    a = ap.parse_args()
    train(a) if a.mode == "train" else evaluate(a)
