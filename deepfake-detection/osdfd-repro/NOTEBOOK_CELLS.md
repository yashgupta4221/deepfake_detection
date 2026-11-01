# Kaggle notebook (Add Input: gend-official-data, celeb-df-v2 [reubensuju], osdfd-repro; GPU on; Internet on) -> Run All
# CELL 1 - setup
import os, glob
!pip install -q lz4
!pip install -q facenet-pytorch --no-deps
CODE=os.path.dirname(glob.glob("/kaggle/input/**/train_osdfd.py",recursive=True)[0])
FF=glob.glob("/kaggle/input/**/FF-x1.3-th0.5-all.tar.lz4",recursive=True)[0]
DF=glob.glob("/kaggle/input/**/DFDC-pre-x1.3-th0.5-all.tar.lz4",recursive=True)[0]
CDF=os.path.dirname(glob.glob("/kaggle/input/**/List_of_testing_videos.txt",recursive=True)[0])
TMP="/kaggle/temp" if os.path.isdir("/kaggle/temp") else "/tmp"; OUT="/kaggle/working"
K=12; MINUTES=45
os.makedirs(f"{TMP}/cache",exist_ok=True)
print(CODE,FF,DF,CDF,TMP,sep="\n"); !df -h {TMP} | tail -1; !nvidia-smi | head -12
# CELL 2 - FF++ training frames
!python {CODE}/extract_subset.py --archive {FF} --kind ffpp --out {TMP}/crops/ffpp --max_real 7000
# CELL 3 - cache frozen-prefix tokens
!python {CODE}/cache_tokens.py --manifest {TMP}/crops/ffpp/manifest.csv --out {TMP}/cache/ffpp --K {K}
# CELL 4 - 40-iteration smoke test (catches errors / shows speed)
!python {CODE}/train_osdfd.py --cache {TMP}/cache/ffpp --out {OUT}/smoke --max_iters 40 --eval_every 40
# CELL 5 - real training (time budget)
!python {CODE}/train_osdfd.py --cache {TMP}/cache/ffpp --out {OUT}/run --minutes {MINUTES}
# CELL 6 - DFDC test
!python {CODE}/extract_subset.py --archive {DF} --kind dfdc --out {TMP}/crops/dfdc --n_test 20000
!python {CODE}/eval_osdfd.py --ckpt {OUT}/run/osdfd_last.pt --sets dfdc={TMP}/crops/dfdc/manifest.csv --out {OUT}/results_dfdc.json
# CELL 7 - Celeb-DF v2 test (official 518 videos, x1.3 crops)
!python {CODE}/cdf_crops.py --root {CDF} --out {TMP}/crops/cdf2
!python {CODE}/eval_osdfd.py --ckpt {OUT}/run/osdfd_last.pt --sets cdf2={TMP}/crops/cdf2/manifest.csv --out {OUT}/results_cdf2.json
# CELL 8 - results
import json
for f in ["results_dfdc.json","results_cdf2.json"]: print(f, json.load(open(f"{OUT}/{f}")))
