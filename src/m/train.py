import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from metrics import classification_metrics, confusion_matrix
from utils import save_checkpoint, write_csv, save_metrics_summary, plot_and_save_confusion
import numpy as np

def train_one_config(model, device, loaders, criterion, optimizer, config_pair, out_dirs, num_epochs):
    """
    Train model for a single config and return test metrics.
    """
    train_loader, val_loader, test_loader = loaders
    scaler = GradScaler()  # safe for all PyTorch versions

    # Scheduler: CosineAnnealingLR
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = []
    best_val_acc = 0.0
    best_ckpt = None

    for epoch in range(num_epochs):
        # --------- TRAIN ---------
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [train]", unit='batch')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = torch.argmax(outputs.detach(), dim=1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            total += labels.size(0)
            pbar.set_postfix({'loss': f"{running_loss/total:.4f}", 'acc': f"{running_corrects/total:.4f}"})

        epoch_loss = running_loss / total
        epoch_acc = running_corrects / total

        # --------- VALIDATION ---------
        model.eval()
        val_loss, val_corrects, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [val]", unit='batch'):
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data).item()
                val_total += labels.size(0)

        val_loss /= max(val_total,1)
        val_acc = val_corrects / max(val_total,1)

        # Scheduler step
        scheduler.step()

        # Save history
        history.append((epoch+1, epoch_loss, epoch_acc, val_loss, val_acc))

        # Save checkpoint each epoch
        ckpt_name = f"bs{config_pair[0]}_ep{config_pair[1]}_epoch{epoch+1}"
        save_checkpoint({'model_state': model.state_dict(),
                         'optimizer_state': optimizer.state_dict(),
                         'epoch': epoch+1}, out_dirs['checkpoints'], ckpt_name)

        # Print epoch summary
        print(f"[Epoch {epoch+1}] train_loss={epoch_loss:.4f}, train_acc={epoch_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_ckpt = out_dirs['checkpoints'] / f"{ckpt_name}.pth"

    # Save training history CSV
    hist_csv = out_dirs['results'] / f"history_bs{config_pair[0]}_ep{config_pair[1]}.csv"
    rows = [('epoch','train_loss','train_acc','val_loss','val_acc')]
    for e, tl, ta, vl, va in history:
        rows.append((e, f"{tl:.6f}", f"{ta:.6f}", f"{vl:.6f}", f"{va:.6f}"))
    write_csv(hist_csv, rows)

    # --------- EVALUATE BEST CKPT ON TEST ---------
    if best_ckpt is not None and best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device)['model_state'])

    # Evaluate on test set
    model.eval()
    all_probs, all_preds, all_targets = [], [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing", unit='batch'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_probs.append(probs)
            all_preds.append(preds)
            all_targets.append(labels.cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_preds = np.hstack(all_preds)
    all_targets = np.hstack(all_targets)

    metrics = classification_metrics(all_targets, all_preds, all_probs)
    cm = confusion_matrix(all_targets, all_preds)

    # Save metrics and confusion matrix
    save_metrics_summary(out_dirs['results'] / f"metrics_bs{config_pair[0]}_ep{config_pair[1]}.json", metrics)
    plot_and_save_confusion(cm, test_loader.dataset.classes, out_dirs['confusion'] / f"confusion_bs{config_pair[0]}_ep{config_pair[1]}.png")

    # Print metrics to terminal
    print("\n=== Test Metrics ===")
    print(f"Confusion Matrix:\n{cm}")
    for k,v in metrics.items():
        print(f"{k}: {v}")

    return metrics
