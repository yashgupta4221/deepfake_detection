"""Mock tests for the fdnet v2 switches (aug / sbi / sched / tta / resume) + a regression check that DEFAULTS reproduce v1 exactly."""
import csv, importlib.util, json, os, tempfile, types
import numpy as np, torch
from PIL import Image
from transformers import CLIPVisionConfig, CLIPVisionModel
import fdnet as fd

spec = importlib.util.spec_from_file_location("fdnet_v1", "fdnet_v1.py"); v1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v1)
cfg = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4, image_size=224, patch_size=14, hidden_act="quick_gelu")
def fresh(): torch.manual_seed(123); return CLIPVisionModel(cfg).eval()

tmp = tempfile.mkdtemp(); rng = np.random.RandomState(0); rows = []
for s in fd.GROUPS:
    for v in range(12):
        for f in range(2):
            p = f"{tmp}/{s}_{v}_{f}.png"; Image.fromarray((rng.rand(64, 64, 3) * 255).astype(np.uint8)).save(p)
            rows.append(dict(path=p, label=0 if s == "real" else 1, source=s, video=f"{s}_{v}"))
with open(f"{tmp}/ff.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["path", "label", "source", "video"]); w.writeheader(); w.writerows(rows)

def ns(out, **k):
    return types.SimpleNamespace(**{**dict(clip="x", out=out, workers=0, fuse="last", grp="none", r=4, alpha=8, manifest=f"{tmp}/ff.csv", seed=0, init_seed=0, split_seed=0, val_frac=0.15,
                                          iters=6, lr=1e-3, eta=0.01, n_real=4, n_per_fake=2, flip=False, eval_every=3, gc="auto"), **k})

# 1) regression: v2 with default flags == v1, bit for bit
v1.train(ns(f"{tmp}/o1"), clip=fresh()); fd.train(ns(f"{tmp}/o2"), clip=fresh())
a, b = torch.load(f"{tmp}/o1/last.pt")["model"], torch.load(f"{tmp}/o2/last.pt")["model"]
assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a); print("1) v2 defaults reproduce v1 weights exactly: ok")

# 2) augmentation / self-blend: same size, valid, actually change the image, blob mask in [0,1] with a real interior and exterior
img = Image.open(rows[0]["path"]).convert("RGB").resize((224, 224)); random_state = __import__("random")
random_state.seed(1); sb = fd.self_blend(img); au = fd.augment(img); m = fd._blob_mask(224, 224)
assert sb.size == au.size == img.size and np.abs(np.asarray(sb, float) - np.asarray(img, float)).mean() > 0.5 and 0 <= m.min() and m.max() <= 1 and m[112, 112] > 0.9 and m[0, 0] < 0.05
print("2) self_blend / augment / blob mask: ok | mean |blend-orig| =", round(float(np.abs(np.asarray(sb, float) - np.asarray(img, float)).mean()), 2))

# 3) dataset: SBI-marked index -> label 1, group 5, regular index untouched
paths = [r["path"] for r in rows]; labels = [int(r["label"]) for r in rows]; gid = [fd.GROUPS.index(r["source"]) for r in rows]
ds = fd.FrameDS(paths, labels, gid, aug="std"); _, y0, g0 = ds[0]; _, y1, g1 = ds[fd.SBI_OFF + 0]
assert (y0, g0) == (0, 0) and (y1, g1) == (1, fd.SBI_ID); print("3) SBI sample label/group: ok")

# 4) sampler: batch composition with SBI
src = np.array([r["source"] for r in rows]); pools = {s: np.where(src == s)[0] for s in fd.GROUPS}; pools["SBI"] = pools["real"]
b = next(iter(fd.InfBatches(pools, 4, 2, 0, 3))); assert len(b) == 4 + 8 + 3 and sum(i >= fd.SBI_OFF for i in b) == 3 and all(src[i - fd.SBI_OFF] == "real" for i in b if i >= fd.SBI_OFF)
b0 = next(iter(fd.InfBatches(pools, 4, 2, 0, 0))); assert len(b0) == 12 and all(i < fd.SBI_OFF for i in b0); print("4) sampler composition: ok")

# 5) full v2 recipe trains (aug + sbi + cosine + dro with 6 groups + multi-fusion), evaluates with flip TTA and soup
out = f"{tmp}/o3"; fd.train(ns(out, aug="std", sbi=3, sched="cos", warmup=2, grp="dro", fuse="multi", iters=6), clip=fresh())
d = json.load(open(f"{out}/DONE.json")); assert d["iters"] == 6 and d["resumed"] is False; print("5) full v2 training: ok")
model = fd.Detector(fresh(), r=4, alpha=8, fuse="last"); model.eval(); rs = [r for r in rows if r["source"] in ("real", "DF")][:10]
p0 = fd.score_set(model, rs, 4, 0, "none"); pf = fd.score_set(model, rs, 4, 0, "flip")
ds0 = fd.FrameDS([r["path"] for r in rs], [0] * len(rs), [0] * len(rs)); xs = torch.stack([ds0[i][0] for i in range(len(rs))])
with torch.no_grad(): ref = (torch.sigmoid(model(xs)) + torch.sigmoid(model(torch.flip(xs, dims=[3])))) / 2
assert np.allclose(pf, ref.numpy(), atol=3e-2) and np.abs(p0 - pf).max() > 0; print("   flip-TTA equals mean of original and mirrored: ok")

# 6) group loss with 6 groups starts equal to the plain loss
torch.manual_seed(0); lg = torch.randn(15); y = torch.tensor([0.] * 4 + [1.] * 8 + [1.] * 3); g = torch.tensor([0] * 4 + [1] * 2 + [2] * 2 + [3] * 2 + [4] * 2 + [5] * 3)
q0 = torch.tensor([4 / 15] + [2 / 15] * 4 + [3 / 15]); lp, _, _ = fd.group_loss(lg, y, g, q0, False, 0.01); ld, q1, _ = fd.group_loss(lg, y, g, q0, True, 0.01)
assert abs(float(lp) - float(ld)) < 1e-6 and q1.numel() == 6 and abs(float(q1.sum()) - 1) < 1e-6; print("6) 6-group DRO starts identical to plain: ok")

# 7) crash resume: kill after the it-3 snapshot, restart with --resume
out = f"{tmp}/o4"; fd.train(ns(out, resume_every=3, iters=6), clip=fresh()); assert os.path.exists(f"{out}/resume.pt")
os.remove(f"{out}/DONE.json"); os.remove(f"{out}/last.pt"); fd.train(ns(out, resume_every=3, iters=6, resume=True), clip=fresh())
d = json.load(open(f"{out}/DONE.json")); ck = torch.load(f"{out}/last.pt"); assert d["resumed"] is True and ck["it"] == 6 and os.path.exists(f"{out}/last.pt"); print("7) resume after a simulated crash: ok")
print("ALL FDNET V2 MOCK TESTS PASSED")
