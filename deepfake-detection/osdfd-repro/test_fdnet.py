"""Mock tests for fdnet.py on a tiny random CLIP (no download, CPU)."""
import csv, json, os, sys, tempfile, types
import numpy as np, torch
from PIL import Image
from transformers import CLIPVisionConfig, CLIPVisionModel
import fdnet as fd

cfg = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4, image_size=224, patch_size=14, hidden_act="quick_gelu")
def fresh(): torch.manual_seed(123); return CLIPVisionModel(cfg).eval()
A = lambda **k: types.SimpleNamespace(**{**dict(r=4, alpha=8, fuse="last", grp="none", init_seed=0, clip="x"), **k})

# 1) at init (LoRA B = 0) the backbone hidden states equal the untouched CLIP
x = torch.randn(3, 3, 224, 224); ref = fresh()(pixel_values=x, output_hidden_states=True).hidden_states
m = fd.Detector(fresh(), r=4, alpha=8, fuse="multi", layers=(2, 3, 4)); m.eval()
hs = m.clip(pixel_values=x, output_hidden_states=True).hidden_states
print("1) max |hidden - CLIP| at init:", max((a - b).abs().max().item() for a, b in zip(hs, ref))); assert all((a - b).abs().max() < 1e-5 for a, b in zip(hs, ref))

# 2) only LoRA + head train; no CLIP weight does
tr = [n for n, p in m.named_parameters() if p.requires_grad]; bad = [n for n in tr if n.startswith("clip.") and not (n.endswith(".A") or n.endswith(".B"))]
print("2) trainable tensors:", len(tr), "| CLIP base weights trainable:", bad); assert not bad
assert all(p.requires_grad is False for n, p in m.named_parameters() if n.endswith("base.weight"))
assert set(m.trainable_state()) == set(tr), (set(m.trainable_state()) ^ set(tr))

# 3) same init_seed -> identical init; different init_seed -> different
s0 = fd.Detector(fresh(), r=4, fuse="multi", layers=(2, 3, 4), init_seed=0).trainable_state(); s0b = fd.Detector(fresh(), r=4, fuse="multi", layers=(2, 3, 4), init_seed=0).trainable_state()
s1 = fd.Detector(fresh(), r=4, fuse="multi", layers=(2, 3, 4), init_seed=1).trainable_state()
assert all(torch.equal(s0[k], s0b[k]) for k in s0) and any(not torch.equal(s0[k], s1[k]) for k in s0); print("3) init_seed reproducible & distinct: ok")

# 4) group loss: dro starts identical to plain; q stays a distribution; gradients flow
torch.manual_seed(0); logit = torch.randn(48, requires_grad=True); y = torch.tensor([0.] * 24 + [1.] * 24); g = torch.tensor([0] * 24 + [1] * 6 + [2] * 6 + [3] * 6 + [4] * 6)
q0 = torch.tensor([24 / 48] + [6 / 48] * 4); lp, _, _ = fd.group_loss(logit, y, g, q0, False, 0.01); ld, q1, gl = fd.group_loss(logit, y, g, q0, True, 0.01)
print("4) plain", float(lp), "dro@init", float(ld), "| q sums to", float(q1.sum())); assert abs(float(lp) - float(ld)) < 1e-6 and abs(float(q1.sum()) - 1) < 1e-6
q = q0
for _ in range(2000): _, q, _ = fd.group_loss(logit, y, g, q, True, 0.05)
assert float(q.min()) >= 0.0199 and abs(float(q.sum()) - 1) < 1e-5; print("   q after 2000 steps:", [round(float(v), 3) for v in q])

# 5) partitions are deterministic and disjoint
vids = [f"v{i}" for i in range(3000)]; p = [fd.part_of(v) for v in vids]; assert p == [fd.part_of(v) for v in vids]; print("5) dev share:", round(p.count("dev") / len(p), 3))

# 6) soup math
a, b = {"w": torch.ones(2)}, {"w": torch.ones(2) * 3}; assert torch.allclose(fd.merge_states([a, b])["w"], torch.ones(2) * 2); print("6) soup average: ok")

# 7) end-to-end mock: tiny images, train 6 its for each fuse/grp combo, eval none / soup / ens
tmp = tempfile.mkdtemp(); rng = np.random.RandomState(0)
def mk(prefix, sources, n_vid):
    rows = []
    for si, s in enumerate(sources):
        for v in range(n_vid):
            for f in range(2):
                pth = f"{tmp}/{prefix}_{s}_{v}_{f}.png"; Image.fromarray((rng.rand(32, 32, 3) * 255).astype(np.uint8)).save(pth)
                rows.append(dict(path=pth, label=0 if s == "real" else 1, source=s, video=f"{prefix}_{s}_{v}"))
    with open(f"{tmp}/{prefix}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "label", "source", "video"]); w.writeheader(); w.writerows(rows)
mk("ff", fd.GROUPS, 12); mk("test", ["real", "fake"], 8)
for fuse in ("last", "multi"):
    for grp in ("none", "dro"):
        out = f"{tmp}/run_{fuse}_{grp}"
        a = types.SimpleNamespace(clip="x", out=out, workers=0, fuse=fuse, grp=grp, r=4, alpha=8, manifest=f"{tmp}/ff.csv", seed=0, init_seed=0, split_seed=0, val_frac=0.15,
                                  iters=6, lr=1e-3, eta=0.01, n_real=4, n_per_fake=2, flip=False, eval_every=3, gc="auto")
        # tiny model with fused layers capped to the 4 layers the mock has
        fd.train(a, clip=fresh())
        assert os.path.exists(f"{out}/last.pt") and os.path.exists(f"{out}/DONE.json")
# a second seed for the merge tests (same init_seed, different data order)
a2 = types.SimpleNamespace(clip="x", out=f"{tmp}/run_last_none_s1", workers=0, fuse="last", grp="none", r=4, alpha=8, manifest=f"{tmp}/ff.csv", seed=1, init_seed=0, split_seed=0, val_frac=0.15,
                           iters=6, lr=1e-3, eta=0.01, n_real=4, n_per_fake=2, flip=False, eval_every=3, gc="auto")
fd.train(a2, clip=fresh())
ck = [f"{tmp}/run_last_none/last.pt", f"{tmp}/run_last_none_s1/last.pt"]
for merge, cks in (("none", ck[:1]), ("soup", ck), ("ens", ck)):
    e = types.SimpleNamespace(clip="x", ckpt=cks, merge=merge, fuse="last", grp="none", r=4, alpha=8, init_seed=0, bs=8, workers=0, split_seed=0, val_frac=0.15, out=f"{tmp}/res_{merge}.json",
                              save_scores=f"{tmp}/scores_{merge}", sets=[f"toy_dev={tmp}/test.csv:dev", f"toy_final={tmp}/test.csv:final", f"toy_all={tmp}/test.csv", f"ffval={tmp}/ff.csv:ffval"])
    try:
        fd.evaluate(e, clip=fresh()); r = json.load(open(e.out)); print("7) eval", merge, {k: (v["n_videos"], round(v["frame_auroc"], 3)) for k, v in r.items()})
    except ValueError as ex:      # a tiny partition can contain a single class; that is a property of the toy data, not the code
        print("7) eval", merge, "toy partition had one class:", ex)
# dev and final partitions together cover 'all' exactly
rd = [r for r in fd.read_manifest(f"{tmp}/test.csv") if fd.part_of(r["video"]) == "dev"]; rf = [r for r in fd.read_manifest(f"{tmp}/test.csv") if fd.part_of(r["video"]) == "final"]
assert len(rd) + len(rf) == len(fd.read_manifest(f"{tmp}/test.csv")); print("   dev+final = all:", len(rd), "+", len(rf))
rows_ff = fd.read_manifest(f"{tmp}/ff.csv"); hv = fd.heldout_videos(rows_ff, 0, 0.15)
tr_vids = {r["video"] for r in rows_ff} - hv
assert hv and not (hv & tr_vids) and len(hv) == max(1, int(len({r["video"] for r in rows_ff}) * 0.15)); print("8) held-out FF++ videos:", len(hv), "disjoint from the", len(tr_vids), "training videos")
# 9) gradient checkpointing gives the same gradients as the plain path
def grads(gc):
    mm = fd.Detector(fresh(), r=4, alpha=8, fuse="multi", layers=(2, 3, 4)); mm.train(); mm.gc = gc
    for n, p in mm.named_parameters():
        if n.endswith(".B"): p.data.normal_(0, 0.05)
    torch.manual_seed(5); xx = torch.randn(4, 3, 224, 224); lg = mm(xx); lg.sum().backward()
    return {n: p.grad.clone() for n, p in mm.named_parameters() if p.grad is not None}
g0, g1 = grads(False), grads(True); assert g0.keys() == g1.keys() and len(g0) > 0
print("9) gradient-checkpointing max grad diff:", max((g0[k] - g1[k]).abs().max().item() for k in g0)); assert all(torch.allclose(g0[k], g1[k], atol=1e-5) for k in g0)
print("ALL FDNET MOCK TESTS PASSED")
