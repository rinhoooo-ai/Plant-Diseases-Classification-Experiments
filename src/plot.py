import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_history(csv_path: Path, out_dir: Path):
    """
    Plot training history for a single config.
    Generates two line plots: Loss (train vs val) and Accuracy (train vs val).

    Args:
        csv_path: Path to CSV file of history (history_bs{batch}_ep{epoch}.csv)
        out_dir: Directory to save plots
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)

    # --- Plot Loss ---
    plt.figure(figsize=(8,5))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Val Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f'{csv_path.stem}_loss.png')
    plt.close()

    # --- Plot Accuracy ---
    plt.figure(figsize=(8,5))
    plt.plot(df['epoch'], df['train_acc'], label='Train Accuracy', marker='o')
    plt.plot(df['epoch'], df['val_acc'], label='Val Accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train vs Val Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f'{csv_path.stem}_acc.png')
    plt.close()


def plot_test(csv_path: Path, out_dir: Path):
    """
    Plot test performance across multiple configs.
    X-axis: config labels (batch_size-epochs)
    Y-axis: metric value (0-1)
    Each metric is plotted as a separate line.

    Args:
        csv_path: Path to CSV file containing test performance across configs
        out_dir: Directory to save the plot
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)

    # --- Prepare X-axis labels ---
    df['config_label'] = df['bs'].astype(str) + '-' + df['epochs'].astype(str)
    x = df['config_label'].tolist()

    # List of metrics to plot
    metrics = ['accuracy','top1','top3','top5','precision','recall','specificity','f1','auc_macro','kappa']

    plt.figure(figsize=(12,6))
    for metric in metrics:
        if metric in df.columns:
            plt.plot(x, df[metric], label=metric, marker='o')

    # --- Plot formatting ---
    plt.xlabel('Config (batch-epochs)')
    plt.ylabel('Metric Value')
    plt.ylim(0,1)  # all metrics normalized to 0-1
    plt.title('Test Performance across Configs')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_dir / 'test_performance_across_configs.png')
    plt.close()
