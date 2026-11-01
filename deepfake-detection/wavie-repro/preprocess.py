"""Face-frame extraction following the paper (Sec. IV-A): MTCNN (facenet_pytorch), largest face per frame,
box expanded by 20 px (only when the expanded box stays in-bounds), crop, resize to 224x224, drop frames
with no face. FF++ training: real videos sampled at 1 fps, fake videos at 4 fps.
Writes PNG crops + a manifest CSV (path,label,source,video).

  FF++ (raw / c0):  python preprocess.py ffpp --root /data/FF++ --out /data/crops/ffpp
  Celeb-DF v1/v2:   python preprocess.py videos --real DIR [DIR ...] --fake DIR [DIR ...] --fps 1 --out ...
  WildDeepFake:     python preprocess.py images --real DIR --fake DIR --out ...   (frames already images)
"""
import argparse, csv, glob, os, cv2, numpy as np, torch
from facenet_pytorch import MTCNN

FF_FAKE = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
dev = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(keep_all=True, device=dev, post_process=False)


def crop_face(img_rgb, margin=20, size=224):
    boxes, _ = mtcnn.detect(img_rgb)
    if boxes is None or len(boxes) == 0:
        return None
    b = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))        # largest face
    H, W = img_rgb.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in b]
    ex = (x1 - margin, y1 - margin, x2 + margin, y2 + margin)
    if ex[0] >= 0 and ex[1] >= 0 and ex[2] <= W and ex[3] <= H:        # expand only when in-bounds
        x1, y1, x2, y2 = ex
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
    if x2 <= x1 or y2 <= y1:
        return None
    return cv2.resize(img_rgb[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_AREA)


def video_frames(path, fps):
    cap = cv2.VideoCapture(path); vf = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(vf / fps)), 1); i = 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        if i % step == 0: yield i, cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()


def run(items, out, fps_of, writer):
    """items: list of (video_or_image_path, label, source)"""
    for n, (p, label, src) in enumerate(items):
        vid = os.path.splitext(os.path.basename(p))[0]
        d = os.path.join(out, src); os.makedirs(d, exist_ok=True)
        if p.lower().endswith((".png", ".jpg", ".jpeg")):
            frames = [(0, cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))]
        else:
            frames = video_frames(p, fps_of(label))
        for i, fr in frames:
            c = crop_face(fr)
            if c is None: continue
            f = os.path.join(d, f"{vid}_{i:06d}.png")
            cv2.imwrite(f, cv2.cvtColor(c, cv2.COLOR_RGB2BGR))
            writer.writerow([f, label, src, vid])
        if n % 50 == 0: print(n, "/", len(items), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("kind", choices=["ffpp", "videos", "images"])
    ap.add_argument("--root"); ap.add_argument("--real", nargs="*", default=[]); ap.add_argument("--fake", nargs="*", default=[])
    ap.add_argument("--out", required=True); ap.add_argument("--fps", type=float, default=1.0)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if a.kind == "ffpp":
        items = [(p, 0, "real") for p in sorted(glob.glob(f"{a.root}/original_sequences/youtube/raw/videos/*.mp4"))]
        for m in FF_FAKE:
            items += [(p, 1, m) for p in sorted(glob.glob(f"{a.root}/manipulated_sequences/{m}/raw/videos/*.mp4"))]
        fps_of = lambda label: 1.0 if label == 0 else 4.0              # real 1 fps, fake 4 fps
    else:
        ext = ("*.mp4", "*.avi") if a.kind == "videos" else ("*.png", "*.jpg", "*.jpeg")
        items = []
        for lab, dirs in ((0, a.real), (1, a.fake)):
            for d in dirs:
                for e in ext: items += [(p, lab, os.path.basename(d.rstrip("/"))) for p in sorted(glob.glob(f"{d}/**/{e}", recursive=True))]
        fps_of = lambda label: a.fps
    with open(f"{a.out}/manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["path", "label", "source", "video"])
        run(items, a.out, fps_of, w)
