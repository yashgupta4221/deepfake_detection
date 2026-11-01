import json, os, subprocess, sys, glob, zipfile, time, threading
C = json.load(open(sys.argv[1])); OUT, FS, TMP = C["OUT"], C["FS"], C["TMP"]; PY = C.get("PY", "python")
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def backup():
    with zipfile.ZipFile(f"{OUT}/results_only.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in glob.glob(f"{FS}/res_*.json") + glob.glob(f"{FS}/*_s[0-9]/*.json") + glob.glob(f"{FS}/*_s[0-9]/*.log") + glob.glob(f"{FS}/final_table.txt"): z.write(p, os.path.relpath(p, OUT))
# 1) restore the old backup (A0 s0/s1, C0, C1 checkpoints + results)
if os.path.exists(f"{OUT}/fd_backup.zip") and not os.path.exists(f"{FS}/A0_s0/last.pt"):
    zipfile.ZipFile(f"{OUT}/fd_backup.zip").extractall(OUT); log("restored fd_backup.zip")
RUNS = [r for r in ("A0_s0", "A0_s1", "C1_s0") if os.path.exists(f"{FS}/{r}/last.pt")]
log("checkpoints found:", RUNS)
if len(RUNS) < 3: log("WARNING: not all 3 checkpoints present")
# 2) face crops (same commands/arguments as the original Cell 2), the three in parallel
M = {k: f"{TMP}/crops/{k}/manifest.csv" for k in ("ffpp", "dfdc", "cdf2")}
cmds = {"ffpp": [PY, f"{OUT}/extract_subset.py", "--archive", C["FF"], "--kind", "ffpp", "--out", f"{TMP}/crops/ffpp", "--max_real", "7000"],
        "dfdc": [PY, f"{OUT}/extract_subset.py", "--archive", C["DF"], "--kind", "dfdc", "--out", f"{TMP}/crops/dfdc", "--n_test", "20000"],
        "cdf2": [PY, f"{OUT}/cdf_crops.py", "--root", C["CDF"], "--out", f"{TMP}/crops/cdf2"]}
procs = {k: subprocess.Popen(c, stdout=open(f"{FS}/crops_{k}.log", "a"), stderr=subprocess.STDOUT) for k, c in cmds.items() if not os.path.exists(M[k])}
for k, p in procs.items(): log("crops", k, "exit code", p.wait())
if not all(os.path.exists(m) for m in M.values()): log("CRASH: a manifest is missing", {k: os.path.exists(m) for k, m in M.items()}); sys.exit(1)
log("crops done"); backup()
# 3) evaluate the three checkpoints on the FINAL sets (two GPUs in parallel)
SETS = [f"cdf2={M['cdf2']}", f"dfdc_final={M['dfdc']}:final", f"dfdc_all={M['dfdc']}", f"ffval={M['ffpp']}:ffval"]
def work(gpu, runs):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    for r in runs:
        res = f"{FS}/res_{r}_final.json"
        if os.path.exists(res): continue
        cmd = [PY, f"{OUT}/fdnet.py", "eval", "--ckpt", f"{FS}/{r}/last.pt", "--merge", "none", "--tta", "none", "--sets"] + SETS + ["--out", res, "--workers", "2"]
        with open(f"{FS}/{r}/eval_final.log", "a") as lg: rc = subprocess.call(cmd, env=env, stdout=lg, stderr=subprocess.STDOUT)
        log("eval", r, "exit code", rc); backup()
ts = [threading.Thread(target=work, args=("0", [r for r in RUNS if r != "A0_s1"])), threading.Thread(target=work, args=("1", [r for r in RUNS if r == "A0_s1"]))]
[t.start() for t in ts]; [t.join() for t in ts]
# 4) table
L = ["run      CelebDF2 frame/video | DFDC-FINAL frame/video | DFDC-ALL frame/video | FF++ heldout frame"]
for r in RUNS:
    p = f"{FS}/res_{r}_final.json"
    if not os.path.exists(p): L.append(f"{r:8s} MISSING"); continue
    d = json.load(open(p)); f = lambda k: f"{d[k]['frame_auroc']:.4f}/{d[k]['video_auroc']:.4f}"
    L.append(f"{r:8s} {f('cdf2')}        | {f('dfdc_final')}          | {f('dfdc_all')}        | {d['ffval']['frame_auroc']:.4f}")
open(f"{FS}/final_table.txt", "w").write("\n".join(L)); backup(); log("\n" + "\n".join(L)); log("PIPELINE DONE")
