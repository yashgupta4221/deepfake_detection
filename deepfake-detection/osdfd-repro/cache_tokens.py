"""Run the FROZEN first K CLIP blocks once and cache the tokens entering block K ([N,257,d] fp16). Exact, because the
prefix is frozen and there is no data augmentation. Usage:
  python cache_tokens.py --manifest crops/ffpp/manifest.csv --out cache/ffpp --K 12"""
import argparse, csv, json, numpy as np, torch
from PIL import Image
from osdfd import OSDFD, MEAN, STD, DEV, autocast, wdtype

M = torch.tensor(MEAN).view(3, 1, 1); S = torch.tensor(STD).view(3, 1, 1)


class DS(torch.utils.data.Dataset):
    def __init__(self, paths): self.p = paths
    def __len__(self): return len(self.p)
    def __getitem__(self, i):
        im = np.asarray(Image.open(self.p[i]).convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
        return (torch.from_numpy(im).permute(2, 0, 1) - M) / S


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=12); ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--clip_name", default="openai/clip-vit-large-patch14"); ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.manifest)))
    paths = [r["path"] for r in rows]
    model = OSDFD(a.clip_name, first_adapt_block=a.K).to(DEV).eval()
    n_tok = (224 // 14) ** 2 + 1
    out = np.lib.format.open_memmap(a.out + "_tok.npy", mode="w+", dtype=np.float16, shape=(len(paths), n_tok, model.d))
    dl = torch.utils.data.DataLoader(DS(paths), batch_size=a.bs, num_workers=a.workers)
    k = 0
    with torch.no_grad(), autocast():
        for x in dl:
            t = model.frozen_prefix(x.to(DEV).to(wdtype(model)))
            out[k:k + len(x)] = t.half().cpu().numpy(); k += len(x)
            if (k // a.bs) % 20 == 0: print(k, "/", len(paths), flush=True)
    out.flush()
    np.save(a.out + "_labels.npy", np.array([int(r["label"]) for r in rows]))
    np.save(a.out + "_sources.npy", np.array([r["source"] for r in rows]))
    np.save(a.out + "_videos.npy", np.array([r["video"] for r in rows]))
    json.dump({"K": a.K, "n": len(paths), "d": model.d}, open(a.out + "_meta.json", "w"))
    print("cached", len(paths), "-> ", a.out + "_tok.npy")
