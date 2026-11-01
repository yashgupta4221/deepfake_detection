import json, re
FD = open("fdnet.py").read()
cells = []
def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t.splitlines(True)})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n").splitlines(True)})

md("""# fdnet v2 - Kaggle notebook (import via File > Import Notebook)
Settings: **Accelerator = GPU T4 x2, Internet ON**. Add Inputs: `gend-official-data` and `celeb-df-v2` ONLY (the two crop scripts are written by Cells 1b and 1c).
Run cells in order. Do NOT change the accelerator after Cell 2 (it wipes the session). Protocol: `PROTOCOL_NEW.md` Addendum C (read-only). Stage 1 only looks at DFDC-DEV and the FF++ held-out set.""")

code('''# CELL 1 - setup
import os, glob
!pip install -q lz4
!pip install -q facenet-pytorch --no-deps
CODE="/kaggle/working"   # extract_subset.py and cdf_crops.py are written by Cells 1b and 1c
FF=glob.glob("/kaggle/input/**/FF-x1.3-th0.5-all.tar.lz4",recursive=True)[0]
DF=glob.glob("/kaggle/input/**/DFDC-pre-x1.3-th0.5-all.tar.lz4",recursive=True)[0]
CDF=os.path.dirname(glob.glob("/kaggle/input/**/List_of_testing_videos.txt",recursive=True)[0])
TMP="/kaggle/temp" if os.path.isdir("/kaggle/temp") else "/tmp"; OUT="/kaggle/working"
FS=f"{OUT}/fd"; os.makedirs(FS,exist_ok=True)
print(CODE,FF,DF,CDF,TMP,sep="\\n")
!nvidia-smi | head -12''')

code("%%writefile /kaggle/working/extract_subset.py\n" + open("extract_subset.py").read().replace("\r", ""))
code("%%writefile /kaggle/working/cdf_crops.py\n" + open("cdf_crops.py").read().replace("\r", ""))
code('''# CELL 2 - face crops (same frames as before). Slow: wait until ALL THREE finish and print their manifest lines
!python {CODE}/extract_subset.py --archive {FF} --kind ffpp --out {TMP}/crops/ffpp --max_real 7000
!python {CODE}/extract_subset.py --archive {DF} --kind dfdc --out {TMP}/crops/dfdc --n_test 20000
!python {CODE}/cdf_crops.py --root {CDF} --out {TMP}/crops/cdf2
FFM=f"{TMP}/crops/ffpp/manifest.csv"; DFM=f"{TMP}/crops/dfdc/manifest.csv"; CDM=f"{TMP}/crops/cdf2/manifest.csv"
print({m: os.path.exists(m) for m in (FFM,DFM,CDM)}); print(sorted(os.listdir(f"{TMP}/crops/ffpp")))''')

code("%%writefile /kaggle/working/fdnet.py\n" + FD)

code('''# CELL 4 - 1-minute smoke test of every v2 switch (needs the GPU; must finish without errors)
FFM=f"{TMP}/crops/ffpp/manifest.csv"; DFM=f"{TMP}/crops/dfdc/manifest.csv"; CDM=f"{TMP}/crops/cdf2/manifest.csv"
for m in (FFM,DFM,CDM): assert os.path.exists(m), m
!rm -rf {TMP}/smoke
!python {OUT}/fdnet.py train --manifest {FFM} --out {TMP}/smoke --iters 20 --eval_every 20 --resume_every 10 --sched cos --warmup 5 --aug std --flip --sbi 12 --workers 2
!python {OUT}/fdnet.py eval --ckpt {TMP}/smoke/last.pt --merge none --sets ffval={FFM}:ffval --tta flip --out {TMP}/smoke/res.json --workers 2
print(open(f"{TMP}/smoke/res.json").read()[:400])''')

code('''# CELL 5 - LOOK at the training images: columns = original | self-blended fake | self-blended + augmentation. Check the blend looks like a face-region mismatch, not garbage.
import sys, random; sys.path.insert(0, OUT); import fdnet as fd
from PIL import Image
real=[r for r in fd.read_manifest(FFM) if r["source"]=="real"]; random.seed(0); tiles=[]
for r in random.sample(real, 6):
    im=Image.open(r["path"]).convert("RGB").resize((224,224)); sb=fd.self_blend(im); tiles += [im, sb, fd.augment(sb)]
canvas=Image.new("RGB",(224*3,224*6))
for k,t in enumerate(tiles): canvas.paste(t,((k%3)*224,(k//3)*224))
canvas.resize((448,896))''')

code('''# CELL 6 - write the queue runner and START STAGE 1 (two background workers, one per GPU). Safe to re-run after a crash: finished jobs are skipped, unfinished ones resume.
import json, subprocess
RUNQ = """
import json, os, subprocess, sys
gpu, jf = sys.argv[1], sys.argv[2]; env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
for job in json.load(open(jf)):
    out = job["out"]; os.makedirs(out, exist_ok=True)
    if not os.path.exists(out + "/DONE.json"):
        with open(out + "/train.log", "a") as lg: rc = subprocess.call(job["train"], env=env, stdout=lg, stderr=subprocess.STDOUT)
        if rc: print("TRAIN FAILED", out, rc, flush=True); continue
    for ev in job["evals"]:
        if not os.path.exists(ev["res"]):
            with open(out + "/eval.log", "a") as lg: subprocess.call(ev["cmd"], env=env, stdout=lg, stderr=subprocess.STDOUT)
"""
open(f"{OUT}/runq.py","w").write(RUNQ)
COS=["--sched","cos","--warmup","100"]; AUG=["--aug","std","--flip"]
ARMS={"A0":(2000,[]), "C0":(6000,COS), "C1":(6000,COS+AUG), "C2":(6000,COS+AUG+["--sbi","12"]), "C3":(6000,COS+AUG+["--sbi","12","--fuse","multi"])}
PY="python"; W="2"
def job(arm, seed, sets, tag, extra_eval=()):
    it, fl = ARMS[arm]; out=f"{FS}/{arm}_s{seed}"; fuse = ["--fuse","multi"] if "multi" in fl else []
    tr=[PY,f"{OUT}/fdnet.py","train","--manifest",FFM,"--out",out,"--seed",str(seed),"--init_seed","0","--iters",str(it),"--eval_every","1000","--workers",W,"--resume"]+fl
    evs=[]
    for tta in ([] if not extra_eval else extra_eval):
        sfx = "" if tta=="none" else "_tta"
        evs.append({"res":f"{FS}/res_{arm}_s{seed}_{tag}{sfx}.json","cmd":[PY,f"{OUT}/fdnet.py","eval","--ckpt",f"{out}/last.pt","--merge","none","--tta",tta,"--sets"]+sets+["--out",f"{FS}/res_{arm}_s{seed}_{tag}{sfx}.json","--workers",W]+fuse})
    return {"out":out,"train":tr,"evals":evs}
DEVSETS=[f"dfdc_dev={DFM}:dev", f"ffval={FFM}:ffval"]
Q0=[job("A0",0,DEVSETS,"dev",("none","flip")), job("C0",0,DEVSETS,"dev",("none","flip")), job("C2",0,DEVSETS,"dev",("none","flip"))]
Q1=[job("A0",1,DEVSETS,"dev",("none","flip")), job("C1",0,DEVSETS,"dev",("none","flip")), job("C3",0,DEVSETS,"dev",("none","flip"))]
json.dump(Q0,open(f"{FS}/q0.json","w")); json.dump(Q1,open(f"{FS}/q1.json","w"))
for g,q in (("0","q0"),("1","q1")):
    subprocess.Popen(["python",f"{OUT}/runq.py",g,f"{FS}/{q}.json"],start_new_session=True,stdout=open(f"{FS}/{q}.out","a"),stderr=subprocess.STDOUT)
print("started 2 workers: GPU0 -> A0_s0, C0_s0, C2_s0 | GPU1 -> A0_s1, C1_s0, C3_s0")''')

code('''# CELL 7 - MONITOR (re-run any time; also keeps the session active). Shows the last log line of each job and GPU use.
import glob, os, time, subprocess
for d in sorted(glob.glob(f"{FS}/*_s[0-9]")):
    lg=f"{d}/train.log"; ln=[l for l in open(lg).read().splitlines() if l.strip()] if os.path.exists(lg) else []
    print(os.path.basename(d), "| DONE" if os.path.exists(f"{d}/DONE.json") else "| running/queued", "|", (ln[-1][:140] if ln else "(no log yet)"))
print(sorted(os.path.basename(p) for p in glob.glob(f"{FS}/res_*.json")))
print(subprocess.run("nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader",shell=True,capture_output=True,text=True).stdout)
for q in ("q0","q1"):
    o=open(f"{FS}/{q}.out").read().strip(); print(q, "->", o[-300:] if o else "no messages (good)")''')

code('''# CELL 8 - BACKUP: run when a result appears and before the quota/session ends, then download the zip from the right-hand file panel (/kaggle/working/fd_backup.zip)
import zipfile, glob, os
with zipfile.ZipFile(f"{OUT}/fd_backup.zip","w",zipfile.ZIP_DEFLATED) as z:
    for p in glob.glob(f"{FS}/res_*.json")+glob.glob(f"{FS}/*_s[0-9]/last.pt")+glob.glob(f"{FS}/*_s[0-9]/*.json")+glob.glob(f"{FS}/*_s[0-9]/*.log")+glob.glob(f"{FS}/q*.json"): z.write(p, os.path.relpath(p,OUT))
print(os.path.getsize(f"{OUT}/fd_backup.zip")/1e6,"MB")''')

code('''# CELL 9 - STAGE 1 TABLE (DFDC-DEV + FF++ held-out only) and the pre-registered ladder choice
import json, os
rows={}
for n in ("A0_s0","A0_s1","C0_s0","C1_s0","C2_s0","C3_s0"):
    for sfx,t in (("","none"),("_tta","flip")):
        p=f"{FS}/res_{n}{'_dev'}{sfx}.json"
        if os.path.exists(p):
            r=json.load(open(p)); d=r["dfdc_dev"]; rows[(n,t)]=((d["frame_auroc"]+d["video_auroc"])/2, d["frame_auroc"], d["video_auroc"], d["frame_acc@0.5"], r["ffval"]["frame_auroc"])
print("run      TTA   DEV-score  DFDC-DEV frame  video   acc@0.5  FF++ heldout AUROC")
for (n,t),v in sorted(rows.items()): print(f"{n:8s} {t:5s} {v[0]:.4f}     {v[1]:.4f}        {v[2]:.4f}  {v[3]:.4f}   {v[4]:.4f}")
if all((c,"none") in rows for c in ("C0_s0","C1_s0","C2_s0","C3_s0")):
    best="C0_s0"
    for c in ("C1_s0","C2_s0","C3_s0"):
        if rows[(c,"none")][0]-rows[(best,"none")][0]>=0.005: best=c
    use_tta = rows.get((best,"flip"),(0,))[0]-rows[(best,"none")][0]>=0.005
    print("LADDER CHOICE:",best,"| use flip-TTA in the final evaluation:",use_tta)
else: print("ladder not decided yet: C0-C3 seed-0 results are not all present")''')

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("fdnet_v2_kaggle.ipynb", "w"), indent=1)
print(len(cells), "cells written")
