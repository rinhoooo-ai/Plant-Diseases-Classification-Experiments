from config import DATASETS, EXPERIMENT_CONFIGS, ROOT
from main import run_for_dataset
from pathlib import Path
import csv, json
import pandas as pd
from plot import plot_test, plot_history

OUT_SUMMARY = Path(ROOT) / 'output_all_summary.csv'
rows = [('dataset','bs','epochs','accuracy','top1','top3','top5',
         'precision','recall','specificity','f1','auc_macro','kappa')]

# --------------------------------------------------
# 1) RUN ALL DATASETS AND COLLECT METRICS
# --------------------------------------------------
for ds_key, ds_path in DATASETS.items():
    print('Running dataset', ds_key)
    out_base = Path(ROOT) / ds_key

    for cfg in EXPERIMENT_CONFIGS:
        print('  config', cfg)
        try:
            perf = run_for_dataset(ds_key, ds_path, cfg, out_base)

            row = [
                ds_key,
                cfg[0], cfg[1],
                perf.get('accuracy', ''),
                perf.get('top1', ''), perf.get('top3', ''), perf.get('top5', ''),
                perf.get('precision', ''), perf.get('recall', ''), perf.get('specificity', ''),
                perf.get('f1', ''), perf.get('auc_macro', ''), perf.get('kappa', '')
            ]
            rows.append(row)
        except Exception as e:
            print('Failed config', cfg, 'for dataset', ds_key, 'error:', e)

# Save the combined summary
with open(OUT_SUMMARY, 'w') as f:
    writer = csv.writer(f)
    for r in rows: writer.writerow(r)

print('All done. Summary at', OUT_SUMMARY)