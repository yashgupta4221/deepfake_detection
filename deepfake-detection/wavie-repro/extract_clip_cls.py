"""Frozen CLIP ViT-L/14 (openai/clip-vit-large-patch14): save the [CLS] token of EACH of the 24 transformer
blocks (width 1024, before the final projection) for every face crop. Run once per dataset; training then only
touches the cached features (the backbone is frozen and the paper uses no augmentation, so this is exact).
Usage: python extract_clip_cls.py --manifest crops/ffpp/manifest.csv --out feats/ffpp
Writes <out>_cls.npy [N,24,1024] float16, <out>_labels.npy and <out>_videos.npy."""
import argparse, csv, numpy as np, torch
from PIL import Image
from transformers import CLIPVisionModel

MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)


class DS(torch.utils.data.Dataset):
    def __init__(self, paths): self.p = paths
    def __len__(self): return len(self.p)
    def __getitem__(self, i):
        im = np.asarray(Image.open(self.p[i]).convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
        return (torch.from_numpy(im).permute(2, 0, 1) - MEAN) / STD


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=128); ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.manifest)))
    paths = [r["path"] for r in rows]; labels = np.array([int(r["label"]) for r in rows], dtype=np.int64)
    dev = torch.device("cuda")
    clip = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).to(dev).eval()
    out = np.lib.format.open_memmap(a.out + "_cls.npy", mode="w+", dtype=np.float16, shape=(len(paths), 24, 1024))
    dl = torch.utils.data.DataLoader(DS(paths), batch_size=a.bs, num_workers=a.workers)
    k = 0
    with torch.no_grad():
        for x in dl:
            hs = clip(pixel_values=x.to(dev).half(), output_hidden_states=True).hidden_states[1:]   # 24 block outputs
            out[k:k + len(x)] = torch.stack([h[:, 0] for h in hs], 1).cpu().numpy(); k += len(x)
            if (k // a.bs) % 50 == 0: print(k, "/", len(paths), flush=True)
    out.flush(); np.save(a.out + "_labels.npy", labels)
    np.save(a.out + "_videos.npy", np.array([r.get("video", "") for r in rows]))
