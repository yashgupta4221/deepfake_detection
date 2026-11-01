"""Score face-crop manifests with a trained OSDFD. Frame-level AUROC (paper's headline metric) + video-level AUROC
(mean frame score per video, as in the paper's video-level table).
  python eval_osdfd.py --ckpt runs/osdfd/osdfd_last.pt --sets dfdc=crops/dfdc/manifest.csv cdf2=crops/cdf2/manifest.csv --out results.json"""
import argparse, csv, json, numpy as np, torch
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from osdfd import OSDFD, DEV, autocast, wdtype
from hybrid import build_model
from cache_tokens import DS

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--bs", type=int, default=96)
    ap.add_argument("--clip_name", default="openai/clip-vit-large-patch14"); ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu")
    model = build_model(ck.get("cfg"), a.clip_name, ck["K"])
    print(model.load_state_dict(ck["model"], strict=False).unexpected_keys, "(unexpected keys: should be empty)")
    model = model.to(DEV).eval(); res = {}
    for spec in a.sets:
        name, man = spec.split("=")
        rows = list(csv.DictReader(open(man))); y = np.array([int(r["label"]) for r in rows]); v = np.array([r["video"] for r in rows])
        dl = torch.utils.data.DataLoader(DS([r["path"] for r in rows]), batch_size=a.bs, num_workers=a.workers)
        ps = []
        with torch.no_grad(), autocast():
            for i, x in enumerate(dl):
                ps.append(torch.sigmoid(model.predict_images(x.to(DEV).to(wdtype(model))).float()).cpu())
                if i % 50 == 0: print(name, i * a.bs, "/", len(rows), flush=True)
        p = torch.cat(ps).numpy()
        vp = defaultdict(list); vy = {}
        for pi, yi, vi in zip(p, y, v): vp[vi].append(pi); vy[vi] = yi
        keys = list(vp)
        res[name] = {"n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum()), "frame_auroc": float(roc_auc_score(y, p)),
                     "frame_acc@0.5": float(((p > 0.5) == y).mean()),
                     "video_auroc": float(roc_auc_score([vy[k] for k in keys], [np.mean(vp[k]) for k in keys])), "n_videos": len(keys)}
        print(name, res[name], flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
