"""Stream a .tar.lz4 archive and write ONLY the frames we sample (saves disk), plus a manifest.
Two passes over the archive (lz4 decompression is fast): pass 1 lists the PNG members, we sample with a fixed seed,
pass 2 extracts just the sampled members. Archive layout expected: .../<class>/<video>/<frame>.png with
class in {real, DF, F2F, FS, NT} (FF++) or {real, fake} (DFDC); any leading folder name is ignored.

FF++ : python extract_subset.py --archive FF-x1.3-th0.5-all.tar.lz4 --kind ffpp --out /kaggle/temp/crops/ffpp
DFDC : python extract_subset.py --archive DFDC-pre-x1.3-th0.5-all.tar.lz4 --kind dfdc --out /kaggle/temp/crops/dfdc --n_test 20000
Paper-like training make-up: all real frames up to --max_real, fakes drawn equally from the 4 methods so fake:real = --fake_ratio."""
import argparse, csv, os, random, tarfile
import lz4.frame

CLASSES = {"ffpp": ["real", "DF", "F2F", "FS", "NT"], "dfdc": ["real", "fake"]}


def stream(path):
    return tarfile.open(fileobj=lz4.frame.open(path, "rb"), mode="r|")


def parse(name, classes):
    parts = name.replace("\\", "/").split("/")
    for i, p in enumerate(parts):
        if p in classes and len(parts) - i >= 3:
            return p, parts[i + 1]                     # (class, video)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True); ap.add_argument("--kind", choices=["ffpp", "dfdc"], required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_real", type=int, default=131155); ap.add_argument("--fake_ratio", type=float, default=175465 / 131155)
    ap.add_argument("--n_test", type=int, default=20000, help="dfdc: max frames per class")
    a = ap.parse_args(); random.seed(a.seed); classes = CLASSES[a.kind]

    print("pass 1: listing", flush=True)
    items = {c: [] for c in classes}                                       # class -> [(member_name, video)]
    with stream(a.archive) as tf:
        for m in tf:
            if m.isfile() and m.name.lower().endswith(".png"):
                r = parse(m.name, classes)
                if r: items[r[0]].append((m.name, r[1]))
    print({c: len(v) for c, v in items.items()}, flush=True)

    sel = []                                                                # (member_name, label, class, video)
    for c in classes: random.shuffle(items[c])
    if a.kind == "ffpp":
        real = items["real"][: a.max_real]; per = int(len(real) * a.fake_ratio) // 4
        sel += [(n, 0, "real", v) for n, v in real]
        for c in classes[1:]: sel += [(n, 1, c, v) for n, v in items[c][:per]]
    else:
        sel += [(n, 0, "real", v) for n, v in items["real"][: a.n_test]]
        sel += [(n, 1, "fake", v) for n, v in items["fake"][: a.n_test]]
    want = {n: (lab, c, v) for n, lab, c, v in sel}
    print("selected", len(want), flush=True)

    os.makedirs(a.out, exist_ok=True); rows = []
    print("pass 2: extracting", flush=True)
    with stream(a.archive) as tf:
        for m in tf:
            if m.name in want:
                lab, c, v = want[m.name]
                dst = os.path.join(a.out, c, v); os.makedirs(dst, exist_ok=True)
                f = os.path.join(dst, os.path.basename(m.name))
                with open(f, "wb") as fh: fh.write(tf.extractfile(m).read())
                rows.append((f, lab, c, f"{c}/{v}"))
                if len(rows) % 20000 == 0: print(len(rows), "/", len(want), flush=True)
    with open(os.path.join(a.out, "manifest.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["path", "label", "source", "video"]); w.writerows(rows)
    nr = sum(r[1] == 0 for r in rows)
    print("manifest:", len(rows), "frames | real", nr, "fake", len(rows) - nr, "| videos", len({r[3] for r in rows}))


if __name__ == "__main__":
    main()
