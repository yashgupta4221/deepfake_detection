# FS-VFM baseline reproduction cells (Kaggle: GPU on, Internet ON). Add Inputs: gend-official-data, celeb-df-v2, hybrid-repro (code dataset).
# CELL F1 - setup
import os, glob
!pip install -q lz4
!pip install -q facenet-pytorch --no-deps
!pip install -q timm==0.4.5 huggingface_hub
CODE=os.path.dirname(glob.glob("/kaggle/input/**/hybrid.py",recursive=True)[0])
FF=glob.glob("/kaggle/input/**/FF-x1.3-th0.5-all.tar.lz4",recursive=True)[0]
DF=glob.glob("/kaggle/input/**/DFDC-pre-x1.3-th0.5-all.tar.lz4",recursive=True)[0]
CDF=os.path.dirname(glob.glob("/kaggle/input/**/List_of_testing_videos.txt",recursive=True)[0])
TMP="/kaggle/temp" if os.path.isdir("/kaggle/temp") else "/tmp"; OUT="/kaggle/working"
FS=f"{OUT}/fsvfm_run"; os.makedirs(FS,exist_ok=True)
print(CODE,FF,DF,CDF,TMP,sep="\n")
!nvidia-smi | head -12
# CELL F2 - write the two new scripts (fsvfm_splits.py, fsvfm_eval.py)
import base64, io, tarfile
PAY = "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"
os.makedirs(f"{OUT}/fsvfm_code", exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAY)), mode="r:gz").extractall(f"{OUT}/fsvfm_code")
FC=f"{OUT}/fsvfm_code"; print(os.listdir(FC))
# CELL F3 - the same frames/crops as before (FF++ train; DFDC + Celeb-DF v2 test)
!python {CODE}/extract_subset.py --archive {FF} --kind ffpp --out {TMP}/crops/ffpp --max_real 7000
!python {CODE}/extract_subset.py --archive {DF} --kind dfdc --out {TMP}/crops/dfdc --n_test 20000
!python {CODE}/cdf_crops.py --root {CDF} --out {TMP}/crops/cdf2
# CELL F4 - official FS-VFM code (FSVFM-extension branch) + ViT-B weights from Hugging Face (needs Internet ON)
!git clone -q --depth 1 -b FSVFM-extension https://github.com/wolo-wolo/FSFM-CVPR25 {OUT}/FSFM
from huggingface_hub import hf_hub_download
WD=f"{OUT}/fsvfm_weights"
for fn in ("checkpoint-600.pth","pretrain_ds_mean_std.txt"):
    print(hf_hub_download(repo_id="Wolowolo/fsfm-3c", filename=f"pretrained_models/FS-VFM_ViT-B_VF2_600e/{fn}", local_dir=WD))
PRE=f"{WD}/pretrained_models/FS-VFM_ViT-B_VF2_600e"; REPO=f"{OUT}/FSFM/fsvfm"
# CELL F5 - train/val label files from OUR FF++ manifest (video-level split)
!python {FC}/fsvfm_splits.py --manifest {TMP}/crops/ffpp/manifest.csv --out {TMP}/fsvfm_splits
# CELL F6 - fine-tune (official ViT-B recipe from run_DfD-ViT-B.sh; accum_iter 2 keeps effective batch 64 on one GPU). Progress is printed from the log.
import subprocess, time, re
SEEDS=[0]            # first run ONE seed to check speed and sanity; then set [0,1,2]
for SEED in SEEDS:
    OD=f"{FS}/ft_s{SEED}"
    if glob.glob(f"{OD}/checkpoint-*.pth"): print("skip (done):", OD); continue
    cmd=f"cd {REPO}/finetune/cross_dataset_DFD_and_DiFF && CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python main_finetune_DfD.py --accum_iter 2 --apply_simple_augment --batch_size 32 --nb_classes 2 --model vit_base_patch16 --epochs 10 --blr 2e-4 --layer_decay 0.6 --weight_decay 0.03 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --seed {SEED} --finetune {PRE}/checkpoint-600.pth --train_split {TMP}/fsvfm_splits/train.txt --val_split {TMP}/fsvfm_splits/val.txt --delimiter_in_spilt , --output_dir {OD}"
    os.makedirs(OD, exist_ok=True); LOG=open(f"{OD}/stdout_stderr.txt","w"); p=subprocess.Popen(cmd, shell=True, stdout=LOG, stderr=subprocess.STDOUT, text=True)
    while p.poll() is None:
        time.sleep(120); logs=[f for f in glob.glob(f"{OD}/*.txt") if "stdout_stderr" not in f]
        if logs: print(time.strftime("%H:%M"), open(logs[0]).read()[-400:].replace(chr(10)," | "), flush=True)
    print("exit code", p.returncode, "\n", open(f"{OD}/stdout_stderr.txt").read()[-1500:])
# CELL F7 - evaluate the LAST checkpoint (same metrics/manifests as OSDFD) -> results_FSVFM_s<seed>.json
import json
def last_ckpt(d): return max(glob.glob(f"{d}/checkpoint-*.pth"), key=lambda f: int(re.findall(r"checkpoint-(\d+)", f)[0]))
for SEED in SEEDS:
    OD=f"{FS}/ft_s{SEED}"; ck=last_ckpt(OD); print("evaluating", ck)
    !python {FC}/fsvfm_eval.py --repo {REPO} --ckpt {ck} --model vit_base_patch16 --stats {PRE}/pretrain_ds_mean_std.txt --sets dfdc={TMP}/crops/dfdc/manifest.csv cdf2={TMP}/crops/cdf2/manifest.csv --out {OUT}/results_FSVFM_s{SEED}.json
for f in sorted(glob.glob(f"{OUT}/results_FSVFM_s*.json")): print(os.path.basename(f), json.load(open(f)))

# ---- Hybrid (OSDFD components on FS-VFM): see chat for cells H1-H3 (fsvfm_osdfd.py is written by H1) ----
