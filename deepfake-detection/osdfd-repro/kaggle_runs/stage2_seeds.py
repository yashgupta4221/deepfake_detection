import json, os, subprocess, sys, glob, zipfile, time, threading
import numpy as np
C = json.load(open(sys.argv[1])); OUT, FS, TMP = C["OUT"], C["FS"], C["TMP"]; PY = C.get("PY", "python")
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
M = {k: f"{TMP}/crops/{k}/manifest.csv" for k in ("ffpp", "dfdc", "cdf2")}
SETS = [f"cdf2={M['cdf2']}", f"dfdc_final={M['dfdc']}:final", f"dfdc_all={M['dfdc']}", f"ffval={M['ffpp']}:ffval"]
COS = ["--sched", "cos", "--warmup", "100"]; AUG = ["--aug", "std", "--flip"]
ARMS = {"A0": (2000, []), "C1": (6000, COS + AUG)}
def backup():
    with zipfile.ZipFile(f"{OUT}/results_only.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in glob.glob(f"{FS}/res_*.json") + glob.glob(f"{FS}/*_s[0-9]/*.json") + glob.glob(f"{FS}/*_s[0-9]/*.log") + glob.glob(f"{FS}/*.txt"): z.write(p, os.path.relpath(p, OUT))
    for d in glob.glob(f"{FS}/*_s[0-9]"):
        if os.path.exists(f"{d}/last.pt"):
            with zipfile.ZipFile(f"{OUT}/ckpt_{os.path.basename(d)}.zip", "w", zipfile.ZIP_STORED) as z: z.write(f"{d}/last.pt", os.path.relpath(f"{d}/last.pt", OUT))
def one(gpu, arm, seed):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu); it, fl = ARMS[arm]; out = f"{FS}/{arm}_s{seed}"; os.makedirs(out, exist_ok=True)
    if not os.path.exists(f"{out}/DONE.json"):
        tr = [PY, f"{OUT}/fdnet.py", "train", "--manifest", M["ffpp"], "--out", out, "--seed", str(seed), "--init_seed", "0", "--iters", str(it), "--eval_every", "1000", "--workers", "2", "--resume"] + fl
        log("train start", arm, seed, "gpu", gpu)
        with open(f"{out}/train.log", "a") as lg: rc = subprocess.call(tr, env=env, stdout=lg, stderr=subprocess.STDOUT)
        log("train", arm, seed, "exit code", rc)
        if rc: return
    res = f"{FS}/res_{arm}_s{seed}_final.json"
    if not os.path.exists(res):
        cmd = [PY, f"{OUT}/fdnet.py", "eval", "--ckpt", f"{out}/last.pt", "--merge", "none", "--tta", "none", "--sets"] + SETS + ["--out", res, "--workers", "2"]
        with open(f"{out}/eval_final.log", "a") as lg: rc = subprocess.call(cmd, env=env, stdout=lg, stderr=subprocess.STDOUT)
        log("eval", arm, seed, "exit code", rc)
    backup()
def worker(gpu, jobs):
    for arm, seed in jobs: one(gpu, arm, seed)
ts = [threading.Thread(target=worker, args=("0", [("C1", 1), ("A0", 2)])), threading.Thread(target=worker, args=("1", [("C1", 2)]))]
[t.start() for t in ts]; [t.join() for t in ts]
# table
K = [("cdf2", "frame_auroc"), ("cdf2", "video_auroc"), ("dfdc_all", "frame_auroc"), ("dfdc_all", "video_auroc"), ("dfdc_final", "frame_auroc"), ("dfdc_final", "video_auroc"), ("ffval", "frame_auroc")]
H = "CDF2-frame CDF2-video DFDCall-frame DFDCall-video DFDCfin-frame DFDCfin-video FF++-frame"
L = [f"run      {H}"]; per = {"A0": [], "C1": []}
for arm in ("A0", "C1"):
    for s in (0, 1, 2):
        p = f"{FS}/res_{arm}_s{s}_final.json"
        if not os.path.exists(p): L.append(f"{arm}_s{s}   MISSING"); continue
        d = json.load(open(p)); v = [d[a][b] for a, b in K]; per[arm].append(v); L.append(f"{arm}_s{s}   " + "   ".join(f"{x:.4f}" for x in v))
for arm in ("A0", "C1"):
    if per[arm]:
        a = np.array(per[arm]); L.append(f"{arm} mean " + "   ".join(f"{x:.4f}" for x in a.mean(0))); L.append(f"{arm} std  " + "   ".join(f"{x:.4f}" for x in (a.std(0, ddof=1) if len(a) > 1 else np.zeros(a.shape[1]))))
open(f"{FS}/stage2_table.txt", "w").write("\n".join(L)); backup(); log("\n" + "\n".join(L)); log("STAGE 2 DONE")
