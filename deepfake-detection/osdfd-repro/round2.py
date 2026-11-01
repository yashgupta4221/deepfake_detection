"""Round-2 runner (resumable: finished runs are skipped).  See PROTOCOL.md for the rules, decided BEFORE running.

  Stage A  (selection, FF++ ONLY, NT held out, 2000 its, seed 0):  python round2.py --stage A --code CODE --cache CACHE --out OUT
  Stage B  (final, full FF++, 3432 its, seeds 0-2, test sets):      python round2.py --stage B --arm hyb_image --code CODE --cache CACHE --out OUT \
                                                                         --dfdc <manifest.csv> --cdf2 <manifest.csv>
Every run is trained to a fixed iteration count (--minutes is only a safety cap), so arms are iteration-matched."""
import argparse, os, shlex, subprocess, sys

HYB = ["--arch", "hybrid"]
MOE_ONLY = ["--fsm_p", "0", "--lam", "0"]
ARMS = {                                     # name -> extra train args
    "osdfd":          ["--arch", "osdfd"],
    "hyb_token":      HYB + ["--lora_route", "token"],                 # round-1 'hyb_full'
    "hyb_image":      HYB + ["--lora_route", "image"],
    "hyb_token_moe":  HYB + ["--lora_route", "token"] + MOE_ONLY,      # round-1 'hyb_moe_only'
    "hyb_image_moe":  HYB + ["--lora_route", "image"] + MOE_ONLY,
}
STAGE_A_ARMS = ["osdfd", "hyb_token", "hyb_image", "hyb_token_moe", "hyb_image_moe"]
SEEDS_B = [0, 1, 2]


def sh(cmd):
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def train(a, name, iters, seed, extra):
    out = f"{a.out}/{name}"
    if os.path.exists(f"{out}/DONE.json"):
        print("skip (done):", name, flush=True); return out
    sh([sys.executable, f"{a.code}/train_osdfd.py", "--cache", a.cache, "--out", out, "--max_iters", str(iters), "--minutes", "720",
        "--eval_every", str(a.eval_every), "--seed", str(seed), "--split_seed", "0", "--lr", "1e-4", "--clip_name", a.clip_name]
       + ARMS[extra[0]] + extra[1:] + shlex.split(a.train_extra))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["A", "B"], required=True); ap.add_argument("--code", required=True)
    ap.add_argument("--cache", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="", help="stage B: the hybrid arm chosen by summarize.py")
    ap.add_argument("--dfdc", default=""); ap.add_argument("--cdf2", default="")
    ap.add_argument("--clip_name", default="openai/clip-vit-large-patch14"); ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--train_extra", default="", help="extra args appended to every train_osdfd.py call (used by the mock test)")
    ap.add_argument("--eval_extra", default="")
    ap.add_argument("--iters_a", type=int, default=2000); ap.add_argument("--iters_b", type=int, default=3432)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if a.stage == "A":
        for arm in STAGE_A_ARMS:
            train(a, f"A_{arm}", a.iters_a, 0, [arm, "--holdout", "NT"])
        sh([sys.executable, f"{a.code}/summarize.py", "--out", a.out, "--stage", "A"])
    else:
        assert a.arm in ARMS and a.arm != "osdfd" and a.dfdc and a.cdf2, "--arm <hybrid arm> --dfdc --cdf2 required"
        for seed in SEEDS_B:
            for arm in ("osdfd", a.arm):
                name = f"B_{arm}_s{seed}"; ck = train(a, name, a.iters_b, seed, [arm])
                res = f"{a.out}/results_{name}.json"
                if not os.path.exists(res):
                    sh([sys.executable, f"{a.code}/eval_osdfd.py", "--ckpt", f"{ck}/osdfd_last.pt", "--sets", f"dfdc={a.dfdc}", f"cdf2={a.cdf2}", "--out", res,
                        "--clip_name", a.clip_name] + shlex.split(a.eval_extra))
        sh([sys.executable, f"{a.code}/summarize.py", "--out", a.out, "--stage", "B", "--arm", a.arm])


if __name__ == "__main__":
    main()
