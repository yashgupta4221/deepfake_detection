# Round 2 cells (run in the SAME Kaggle session as round 1, after Cell 1-3 have run; TMP/cache/crops must still exist)
# CELL R0 - paste ROUND2_PATCH_CELL.py (defines CODE2 = patched code folder)
# CELL R1 - Stage A (~3 h): selection on FF++ only, NT held out. Safe to re-run: finished arms are skipped.
R2 = f"{OUT}/round2"
!python {CODE2}/round2.py --stage A --code {CODE2} --cache {TMP}/cache/ffpp --out {R2}
# -> prints the table and "SELECTED hybrid arm: <name>"
# CELL R2 - Stage B (~6 h): replace hyb_image by the selected arm
!python {CODE2}/round2.py --stage B --arm hyb_image --code {CODE2} --cache {TMP}/cache/ffpp --out {R2} --dfdc {TMP}/crops/dfdc/manifest.csv --cdf2 {TMP}/crops/cdf2/manifest.csv
# CELL R3 - save results (small files only)
import shutil, glob
import zipfile
with zipfile.ZipFile("/kaggle/working/round2_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob(f"{R2}/**/*.json", recursive=True): z.write(f, os.path.relpath(f, R2))
