"""Score face-crop manifests with a fine-tuned FS-VFM ViT, using EXACTLY the same metrics/manifests as eval_osdfd.py
(frame AUROC, frame acc@0.5, video AUROC = mean frame score per video), so numbers are directly comparable.
Fake probability = softmax(logits)[:, 1] (our split files use label 1 = fake).
  python fsvfm_eval.py --repo /kaggle/working/FSFM --ckpt .../checkpoint-9.pth --model vit_base_patch16 \
      --stats .../pretrain_ds_mean_std.txt --sets dfdc=crops/dfdc/manifest.csv cdf2=crops/cdf2/manifest.csv --out results.json"""
import argparse, csv, json, os, sys
from collections import defaultdict
import numpy as np, torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import transforms

VGG_MEAN, VGG_STD = [0.5482207536697388, 0.42340534925460815, 0.3654651641845703], [0.2789176106452942, 0.2438540756702423, 0.23493893444538116]


class DS(torch.utils.data.Dataset):          # same eval transform as the official repo: bicubic resize to 224x224 + normalize
    def __init__(self, paths, mean, std, size=224):
        self.p = paths; self.t = transforms.Compose([transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
                                                      transforms.ToTensor(), transforms.Normalize(mean, std)])
    def __len__(self): return len(self.p)
    def __getitem__(self, i): return self.t(Image.open(self.p[i]).convert("RGB"))


def load_stats(path):
    if path and os.path.exists(path):
        d = json.loads(open(path).readline()); return d["mean"], d["std"]
    print("WARNING: no stats file, using VGGFace2 defaults"); return VGG_MEAN, VGG_STD


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--repo", required=True, help="dir containing models_vit.py (…/fsvfm)")
    ap.add_argument("--ckpt", default="", help="fine-tuned checkpoint (empty = random weights, for the mock test only)")
    ap.add_argument("--model", default="vit_base_patch16"); ap.add_argument("--stats", default="")
    ap.add_argument("--sets", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=96); ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args(); sys.path.insert(0, a.repo)
    import models_vit
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = models_vit.__dict__[a.model](num_classes=2, drop_path_rate=0.0, global_pool=True)
    if a.ckpt:
        r = model.load_state_dict(torch.load(a.ckpt, map_location="cpu")["model"], strict=False)
        print("missing:", r.missing_keys, "unexpected:", r.unexpected_keys, "(both should be empty)")
    model = model.to(dev).eval(); mean, std = load_stats(a.stats); res = {}
    for spec in a.sets:
        name, man = spec.split("=")
        rows = list(csv.DictReader(open(man))); y = np.array([int(r["label"]) for r in rows]); v = np.array([r["video"] for r in rows])
        dl = torch.utils.data.DataLoader(DS([r["path"] for r in rows], mean, std), batch_size=a.bs, num_workers=a.workers)
        ps = []
        with torch.no_grad():
            for i, x in enumerate(dl):
                with torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
                    lg = model(x.to(dev))
                ps.append(torch.softmax(lg.float(), -1)[:, 1].cpu())
                if i % 50 == 0: print(name, i * a.bs, "/", len(rows), flush=True)
        p = torch.cat(ps).numpy(); vp = defaultdict(list); vy = {}
        for pi, yi, vi in zip(p, y, v): vp[vi].append(pi); vy[vi] = yi
        keys = list(vp)
        res[name] = {"n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum()), "frame_auroc": float(roc_auc_score(y, p)),
                     "frame_acc@0.5": float(((p > 0.5) == y).mean()),
                     "video_auroc": float(roc_auc_score([vy[k] for k in keys], [np.mean(vp[k]) for k in keys])), "n_videos": len(keys)}
        print(name, res[name], flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
