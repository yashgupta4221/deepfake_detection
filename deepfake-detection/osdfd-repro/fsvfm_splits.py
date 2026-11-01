"""Make FS-VFM train/val label files from OUR FF++ manifest (video-level split, so val frames come from unseen videos).
Format of each line: <absolute path>,<label>   with label 1 = fake, 0 = real (our convention; use --delimiter_in_spilt ',').
  python fsvfm_splits.py --manifest /kaggle/temp/crops/ffpp/manifest.csv --out /kaggle/temp/fsvfm_splits [--val_frac 0.1 --seed 0]"""
import argparse, csv, os, random

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--val_frac", type=float, default=0.1); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rows = list(csv.DictReader(open(a.manifest)))
    vids = sorted({r["video"] for r in rows}); random.Random(a.seed).shuffle(vids)
    val_v = set(vids[:max(1, int(len(vids) * a.val_frac))])
    tr = [r for r in rows if r["video"] not in val_v]; va = [r for r in rows if r["video"] in val_v]
    for name, rs in (("train.txt", tr), ("val.txt", va)):
        with open(os.path.join(a.out, name), "w") as f:
            for r in rs: f.write(f"{os.path.abspath(r['path'])},{int(r['label'])}\n")
        print(name, len(rs), "frames | fake", sum(int(r["label"]) for r in rs), "| videos", len({r["video"] for r in rs}))
