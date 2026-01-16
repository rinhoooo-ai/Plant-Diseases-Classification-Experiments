import torch
from config import PRETRAINED_PATH, DEVICE, DATASETS, LR, WEIGHT_DECAY
from dataset import LeafImageDataset
from model import build_model
from train import train_one_config
from pathlib import Path
import torch.nn as nn
import torch.optim as optim

def run_for_dataset(dataset_key: str, dataset_path: str, config_pair: tuple, out_base: Path):
    """
    Run training & evaluation for a dataset and a config (batch_size, epochs)
    """
    batch_size, epochs = config_pair
    ds = LeafImageDataset(Path(dataset_path), dataset_key, batch_size)
    train_loader, val_loader, test_loader = ds.get_loaders()
    num_classes = ds.get_num_classes()

    model = build_model(num_classes, PRETRAINED_PATH, DEVICE)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    out_dirs = {
        'checkpoints': out_base / 'checkpoints' / f'bs{batch_size}_ep{epochs}',
        'results': out_base / 'results' / f'bs{batch_size}_ep{epochs}',
        'confusion': out_base / 'confusion_matrices' / f'bs{batch_size}_ep{epochs}'
    }
    for p in out_dirs.values():
        p.mkdir(parents=True, exist_ok=True)

    metrics = train_one_config(model, DEVICE,
                               (train_loader, val_loader, test_loader),
                               criterion, optimizer, config_pair, out_dirs, epochs)
    return metrics


if __name__ == "__main__":
    import sys
    dataset_key = sys.argv[1] if len(sys.argv)>1 else 'apple'
    out_base = Path('output') / dataset_key
    perf = run_for_dataset(dataset_key, DATASETS[dataset_key], (16,5), out_base)
    print("Done. Test metrics:", perf)
