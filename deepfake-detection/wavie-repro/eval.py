"""Frame-level AUROC (paper's primary metric) per test set from cached features.
Usage: python eval.py --ckpt runs/wavie/wavie_ema_last.pt --sets cdf1=feats/cdf1 cdf2=feats/cdf2 wdf=feats/wdf [--mode full]"""
import argparse, json, os, numpy as np, torch
from sklearn.metrics import roc_auc_score, accuracy_score
from model import WAVIE
from train import predict

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--sets", nargs="+", required=True)
ap.add_argument("--mode", default="full"); ap.add_argument("--out", default="results.json")
a = ap.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m = WAVIE(mode=a.mode).to(dev); m.load_state_dict(torch.load(a.ckpt, map_location=dev)); m.eval()
res = {}
for s in a.sets:
    name, prefix = s.split("=")
    X = np.load(prefix + "_cls.npy", mmap_mode="r"); y = np.load(prefix + "_labels.npy")
    p = predict(m, X, dev)
    res[name] = {"n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum()),
                 "frame_auroc": float(roc_auc_score(y, p)), "frame_acc@0.5": float(accuracy_score(y, p > 0.5))}
    if os.path.exists(prefix + "_videos.npy"):      # extra, NOT the paper's metric: mean score per video
        v = np.load(prefix + "_videos.npy"); u, inv = np.unique(v, return_inverse=True)
        pv = np.bincount(inv, weights=p) / np.bincount(inv); yv = np.bincount(inv, weights=y) / np.bincount(inv)
        res[name]["video_auroc_extra"] = float(roc_auc_score(yv > 0.5, pv)); res[name]["n_videos"] = int(len(u))
    print(name, res[name])
json.dump(res, open(a.out, "w"), indent=1)
