"""Celeb-DF v2 official test list (518 videos) -> face crops at scale 1.3 (square, centered on the MTCNN box, black padding),
224x224, 1 frame per second. Matches GenD/OSDFD's 'x1.3' crop scale (not the +20px recipe used for WAVIE).
  python cdf_crops.py --root /kaggle/input/datasets/reubensuju/celeb-df-v2 --out /tmp/crops/cdf2"""
import argparse, csv, os, cv2, numpy as np, torch
from facenet_pytorch import MTCNN

dev = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(keep_all=True, device=dev, post_process=False)


def crop_scaled(img, scale=1.3, size=224):
    boxes, _ = mtcnn.detect(img)
    if boxes is None or len(boxes) == 0: return None
    b = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2; s = scale * max(b[2] - b[0], b[3] - b[1]) / 2
    x1, y1, x2, y2 = int(round(cx - s)), int(round(cy - s)), int(round(cx + s)), int(round(cy + s))
    H, W = img.shape[:2]; pad = max(0, -x1, -y1, x2 - W, y2 - H)
    if pad: img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0); x1 += pad; x2 += pad; y1 += pad; y2 += pad
    return cv2.resize(img[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=1.0); ap.add_argument("--scale", type=float, default=1.3)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    items = []
    for line in open(f"{a.root}/List_of_testing_videos.txt"):
        t = line.split()
        if len(t) == 2: items.append((f"{a.root}/{t[1]}", 0 if int(t[0]) == 1 else 1, t[1].split("/")[0]))   # list: 1 = real
    print(len(items), "videos | real", sum(l == 0 for _, l, _ in items), "fake", sum(l == 1 for _, l, _ in items), flush=True)
    rows = []
    for n, (p, lab, src) in enumerate(items):
        vid = os.path.splitext(os.path.basename(p))[0]; d = os.path.join(a.out, src); os.makedirs(d, exist_ok=True)
        cap = cv2.VideoCapture(p); vf = cap.get(cv2.CAP_PROP_FPS) or 30.0; step = max(int(round(vf / a.fps)), 1); i = 0
        while True:
            ok, fr = cap.read()
            if not ok: break
            if i % step == 0:
                c = crop_scaled(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), a.scale)
                if c is not None:
                    f = os.path.join(d, f"{vid}_{i:06d}.png"); cv2.imwrite(f, cv2.cvtColor(c, cv2.COLOR_RGB2BGR))
                    rows.append((f, lab, src, f"{src}/{vid}"))
            i += 1
        cap.release()
        if n % 50 == 0: print(n, "/", len(items), flush=True)
    with open(f"{a.out}/manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["path", "label", "source", "video"]); w.writerows(rows)
    print(len(rows), "face frames")
