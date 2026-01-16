from pathlib import Path
from config import ROOT
from plot import plot_history

ROOT = Path(ROOT)

print("=== Auto plotting history for all datasets ===")

for ds_dir in ROOT.iterdir():
    if not ds_dir.is_dir():
        continue

    results_dir = ds_dir / "results"
    if not results_dir.exists():
        continue

    print(f"Dataset: {ds_dir.name}")

    # folder dạng bs64_ep40
    for cfg_dir in results_dir.iterdir():
        if not cfg_dir.is_dir():
            continue

        # file dạng history_bs64_ep40.csv
        hist_files = list(cfg_dir.glob("history_*.csv"))
        if len(hist_files) == 0:
            continue

        for hist_csv in hist_files:
            print("   Plotting:", hist_csv.name)
            out_dir = ds_dir / "plots_history"
            out_dir.mkdir(exist_ok=True)

            plot_history(hist_csv, out_dir)

print("=== DONE plotting all histories! ===")
