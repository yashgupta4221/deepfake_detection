"""End-to-end smoke test on random images + tiny random CLIP: cache_tokens -> train (osdfd AND hybrid) -> eval (reload from ckpt)."""
import os, sys, csv, subprocess, numpy as np, torch, tempfile
from PIL import Image
from transformers import CLIPVisionConfig, CLIPVisionModel
T = tempfile.mkdtemp(); rng = np.random.RandomState(0)
cfg = CLIPVisionConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=4, image_size=224, patch_size=14, hidden_act="quick_gelu")
CLIPVisionModel(cfg).save_pretrained(f"{T}/clip")
def mk(name, srcs, per_vid=4, n_vid=6):
    rows = []; os.makedirs(f"{T}/{name}", exist_ok=True)
    for s in srcs:
        for v in range(n_vid):
            for f in range(per_vid):
                p = f"{T}/{name}/{s}_{v}_{f}.png"
                Image.fromarray((rng.rand(224, 224, 3) * 255).astype(np.uint8)).save(p)
                rows.append((p, 0 if s == "real" else 1, s, f"{s}/{v}"))
    with open(f"{T}/{name}/manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["path", "label", "source", "video"]); w.writerows(rows)
mk("ffpp", ["real", "DF", "F2F", "FS", "NT"]); mk("test", ["real", "fake"])
run = lambda *a: subprocess.run([sys.executable, *a], check=True, capture_output=True, text=True).stdout
print(run("cache_tokens.py", "--manifest", f"{T}/ffpp/manifest.csv", "--out", f"{T}/cache", "--K", "3", "--clip_name", f"{T}/clip", "--bs", "16", "--workers", "0").splitlines()[-1])
for arch, extra in [("osdfd", []), ("hybrid", ["--arch", "hybrid", "--lora_ranks", "2,4", "--adp_experts", "cv,cd,ad", "--fsm_layers", "3,6"])]:
    o = run("train_osdfd.py", "--cache", f"{T}/cache", "--out", f"{T}/{arch}", "--clip_name", f"{T}/clip", "--max_iters", "30", "--eval_every", "15",
            "--n_real", "6", "--n_per_fake", "2", *extra)
    print(arch, "|", [l for l in o.splitlines() if l.startswith("it 50") or "val AUC" in l or "trainable" in l or l.startswith("finished")])
    r = run("eval_osdfd.py", "--ckpt", f"{T}/{arch}/osdfd_last.pt", "--sets", f"t={T}/test/manifest.csv", "--out", f"{T}/{arch}_res.json", "--clip_name", f"{T}/clip", "--bs", "16", "--workers", "0")
    print(arch, "eval reload ok:", [l for l in r.splitlines() if "unexpected" in l or l.startswith("t {")][0][:120])
# round-2 options: image routing + held-out forgery (NT) + split_seed ; n_per_fake left at default -> auto (6 // 3 = 2)
o = run("train_osdfd.py", "--cache", f"{T}/cache", "--out", f"{T}/hold", "--clip_name", f"{T}/clip", "--max_iters", "30", "--eval_every", "15",
        "--n_real", "6", "--arch", "hybrid", "--lora_ranks", "2,4", "--adp_experts", "cv,cd,ad", "--fsm_layers", "3,6", "--lora_route", "image",
        "--holdout", "NT", "--val_frac", "0.5", "--seed", "3", "--split_seed", "0")
print("holdout |", [l for l in o.splitlines() if "held-out" in l or l.startswith("finished")])
import json; d = json.load(open(f"{T}/hold/DONE.json")); print("DONE.json:", {k: d[k] for k in ("iters", "last_val_auc", "last_holdout_auc")})
assert d["last_holdout_auc"] is not None and d["args"]["n_per_fake"] == 2
r = run("eval_osdfd.py", "--ckpt", f"{T}/hold/osdfd_last.pt", "--sets", f"t={T}/test/manifest.csv", "--out", f"{T}/hold_res.json", "--clip_name", f"{T}/clip", "--bs", "16", "--workers", "0")
print("holdout/image-route eval reload ok:", [l for l in r.splitlines() if "unexpected" in l][0])
print("E2E MOCK OK")
