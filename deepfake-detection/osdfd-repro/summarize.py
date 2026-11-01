"""Summaries with the decision rules from PROTOCOL.md hard-coded (so they cannot be adjusted after seeing results).
  Stage A: held-out-forgery table + the selected hybrid arm.      Stage B: mean +- std over seeds + the verdict."""
import argparse, glob, json, os
import numpy as np

PRIORITY = ["hyb_image", "hyb_token", "hyb_image_moe", "hyb_token_moe"]    # tie-break order: keep FSM+SCL (it is the hybrid), then image routing
TIE = 0.005; STOP_MARGIN = 0.02


def stage_a(out):
    rows = {}
    for d in sorted(glob.glob(f"{out}/A_*/DONE.json")):
        arm = d.split("/")[-2][2:]; r = json.load(open(d)); log = json.load(open(os.path.dirname(d) + "/train_log.json"))
        traj = [round(x["holdout_auc"], 4) for x in log if "holdout_auc" in x]
        rows[arm] = (r["last_holdout_auc"], r["last_val_auc"], traj)
    print("STAGE A - NT held out, FF++ only, last checkpoint, seed 0\n")
    print(f"{'arm':16s} held-out NT AUC | FF++ val AUC (trained forgeries) | held-out AUC at each eval")
    for arm, (h, v, t) in rows.items(): print(f"{arm:16s} {h:.4f}          | {v:.4f}                         | {t}")
    hyb = {k: v[0] for k, v in rows.items() if k in PRIORITY}
    if len(hyb) < len(PRIORITY): print("\nIncomplete: missing", [k for k in PRIORITY if k not in hyb]); return
    best = max(hyb.values()); tied = [k for k in PRIORITY if hyb[k] >= best - TIE]; pick = tied[0]
    print(f"\nSELECTED hybrid arm (rule: highest held-out AUC; ties within {TIE} -> priority order): {pick}   (tied: {tied})")
    if "osdfd" in rows and rows["osdfd"][0] - best > STOP_MARGIN:
        print(f"STOP RULE: OSDFD held-out AUC {rows['osdfd'][0]:.4f} exceeds the best hybrid {best:.4f} by more than {STOP_MARGIN}.")
        print("           The hybrid is not expected to beat OSDFD; Stage B is optional (report this as the finding).")
    print(f"\nNext: python round2.py --stage B --arm {pick} ...")


def stage_b(out, hyb):
    R = {}
    for f in sorted(glob.glob(f"{out}/results_B_*.json")):
        name = os.path.basename(f)[len("results_B_"):-len(".json")]; arm, s = name.rsplit("_s", 1); R.setdefault(arm, []).append(json.load(open(f)))
    its = sorted({json.load(open(d))["iters"] for d in glob.glob(f"{out}/B_*/DONE.json")})
    print(f"STAGE B - full FF++, {'/'.join(map(str, its))} its, last checkpoint, mean +- std (sample, n = seeds) over seeds\n")
    stats = {}
    for arm, lst in R.items():
        print(f"{arm}  (n = {len(lst)} seeds)")
        for ds in ("dfdc", "cdf2"):
            for m in ("frame_auroc", "video_auroc"):
                v = np.array([r[ds][m] for r in lst]); sd = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
                stats[(arm, ds, m)] = (float(v.mean()), sd, v); print(f"   {ds:5s} {m:12s} {v.mean():.4f} +- {sd:.4f}   per seed {np.round(v, 4).tolist()}")
    if "osdfd" in R and hyb in R and len(R["osdfd"]) >= 3 and len(R[hyb]) >= 3:
        print(f"\nVERDICT rule: '{hyb}' counts as better (worse) than OSDFD only if its mean differs by more than the larger seed-std on ALL FOUR metrics.")
        sg = []
        for ds in ("dfdc", "cdf2"):
            for m in ("frame_auroc", "video_auroc"):
                (mo, so, _), (mh, sh_, _) = stats[("osdfd", ds, m)], stats[(hyb, ds, m)]; d = mh - mo; t = max(so, sh_)
                sg.append(1 if d > t else -1 if d < -t else 0); print(f"   {ds:5s} {m:12s} diff {d:+.4f}  threshold {t:.4f}")
        print("   => " + ("HYBRID BETTER on all four" if all(x == 1 for x in sg) else "HYBRID WORSE on all four" if all(x == -1 for x in sg)
                         else "NO RELIABLE DIFFERENCE (mixed or within seed noise)"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--stage", choices=["A", "B"], required=True)
    ap.add_argument("--arm", default=""); a = ap.parse_args()
    stage_a(a.out) if a.stage == "A" else stage_b(a.out, a.arm)
