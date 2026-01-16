import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = "/data/URA/Model Performance Classification/dinov3_ConvNeXt"

# Metrics sẽ vẽ
METRICS = [
    "accuracy", "top1", "top3", "top5",
    "precision", "recall", "f1",
    "specificity", "kappa"
]

def plot_all_metrics(dataset):
    results_dir = Path(ROOT) / dataset / "results"

    if not results_dir.exists():
        print(f"[ERROR] Folder not found: {results_dir}")
        return

    print(f"Processing dataset: {dataset}")
    config_folders = [d for d in results_dir.iterdir() if d.is_dir()]

    configs = []
    metric_values = {m: [] for m in METRICS}

    for cfg_folder in sorted(config_folders):
        cfg_name = cfg_folder.name  # ví dụ: bs64_ep40
        metrics_path = cfg_folder / f"metrics_{cfg_name}.json"

        if not metrics_path.exists():
            print(f"Missing: {metrics_path}")
            continue

        with open(metrics_path, "r") as f:
            data = json.load(f)

        configs.append(cfg_name)

        for m in METRICS:
            val = data.get(m, None)
            # skip NaN
            if isinstance(val, float) and (val != val):
                val = None
            metric_values[m].append(val)

    if not configs:
        print(f"No valid configs found for {dataset}")
        return

    # Plot
    plt.figure(figsize=(12, 6))

    for m in METRICS:
        values = metric_values[m]
        if any(v is not None for v in values):  # vẽ nếu có ít nhất 1 value hợp lệ
            plt.plot(configs, values, marker='o', label=m)

    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.title(f"{dataset} – Performance Across Configs")
    plt.xlabel("Configs")
    plt.ylabel("Metric value")
    plt.grid(True)
    plt.legend()

    out_path = results_dir / f"compare_all_metrics.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"Created plot → {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_compare_configs.py <dataset>")
        sys.exit(1)

    dataset = sys.argv[1]
    plot_all_metrics(dataset)
