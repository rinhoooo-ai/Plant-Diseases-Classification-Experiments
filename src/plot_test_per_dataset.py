# plot_test_per_dataset.py
import pandas as pd
from pathlib import Path
from plot import plot_test

def main():
    ROOT = Path('/data/URA/Model Performance Classification/dinov3_ConvNeXt')

    summary_csv = ROOT / "output_all_summary.csv"
    if not summary_csv.exists():
        print("ERROR: summary file not found:", summary_csv)
        return

    df = pd.read_csv(summary_csv)
    plots_summary = ROOT / "plots_summary"
    plots_summary.mkdir(exist_ok=True)

    # Loop through each dataset and generate its own test performance image
    for ds_key in df['dataset'].unique():
        print(f"Plotting dataset:", ds_key)

        # Extract only rows for this dataset
        df_ds = df[df['dataset'] == ds_key]

        # Save temporary CSV for plot_test()
        tmp_csv = plots_summary / f"{ds_key}_summary_tmp.csv"
        df_ds.to_csv(tmp_csv, index=False)

        # Run plot_test()
        plot_test(tmp_csv, plots_summary)

        # Rename default output → <dataset>.png
        default_png = plots_summary / "test_performance_across_configs.png"
        ds_png = plots_summary / f"{ds_key}.png"

        if default_png.exists():
            default_png.rename(ds_png)
            print(f"✔ Saved → {ds_png}")

        # Remove temp file
        tmp_csv.unlink()

    print("All done! Check plots in:", plots_summary)


if __name__ == "__main__":
    main()
