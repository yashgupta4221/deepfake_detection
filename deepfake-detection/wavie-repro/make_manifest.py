"""Build manifests from the EXISTING pre-cropped GenD releases (no MTCNN step needed):
   FF++ : <ff_root>/{real,DF,F2F,FS,NT}/<video>/*.png      DFDC : <dfdc_root>/{real,fake}/<video>/*.png
Training set mimics the paper's make-up: all real frames (up to --max_real) and fake frames drawn equally from the four
methods so that fake:real = --fake_ratio (paper: 175,465/131,155 = 1.34). Test set: up to --n_test frames per class (random, seed 0).
python make_manifest.py --ff_root /content/data/FF-x1.3-th0.5-all --dfdc_root /content/data/DFDC-pre-x1.3-th0.5-all --out crops"""
import argparse, csv, glob, os, random

def frames(root, dirs):
    out = []
    for d in dirs:
        for f in glob.glob(os.path.join(root, d, "*", "*.png")):
            out.append((f, d, f"{d}/{os.path.basename(os.path.dirname(f))}"))   # (path, source, video id unique per source)
    return out

def write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["path", "label", "source", "video"])
        for p, lab, src, vid in rows: w.writerow([p, lab, src, vid])
    nr = sum(r[1] == 0 for r in rows); print(path, len(rows), "frames | real", nr, "fake", len(rows) - nr, "| videos", len({r[3] for r in rows}))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ff_root", required=True); ap.add_argument("--dfdc_root", required=True); ap.add_argument("--out", default="crops")
    ap.add_argument("--max_real", type=int, default=131155); ap.add_argument("--fake_ratio", type=float, default=175465 / 131155)
    ap.add_argument("--n_test", type=int, default=20000, help="max test frames per class"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); random.seed(a.seed)
    real = frames(a.ff_root, ["real"]); random.shuffle(real); real = real[: a.max_real]
    nfake = int(len(real) * a.fake_ratio); per = nfake // 4; fake = []
    for m in ["DF", "F2F", "FS", "NT"]:
        fm = frames(a.ff_root, [m]); random.shuffle(fm); fake += fm[:per]
    write(f"{a.out}/ffpp/manifest.csv", [(p, 0, s, v) for p, s, v in real] + [(p, 1, s, v) for p, s, v in fake])
    te = []
    for lab, d in ((0, "real"), (1, "fake")):
        fs = frames(a.dfdc_root, [d]); random.shuffle(fs); te += [(p, lab, s, v) for p, s, v in fs[: a.n_test]]
    write(f"{a.out}/dfdc/manifest.csv", te)
