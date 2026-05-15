"""
@author: Giulio Lo Cigno

This is the code for training the models
NOTE: not the full hyperparam space was ever actually executed, we changed the code slightly to start the trainings for the only models we wanted to train
"""
import os
import sys
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from shared_code import (
    SafeImageFolder, collate_skip_none, data_loaders,
    CIFAKE_CNN, I_HAVE_A_THEORY, parse_hparams_from_model_path, GradCAM,
)

DATA_DIR = "./DATA/Cat_dog_splitted/"

BATCH_SIZE = 512
LR = 1e-3
EPOCHS = 30

EARLY_STOP = 3

torch.backends.cudnn.benchmark = True

class EarlyStopping:
    """Stops training when val accuracy has not improved for patience epochs.

    Keeps a deep-copy of the best model weights so training (or interpreting) can be resumed
    from the best checkpoint after stopping.

    Args:
        patience: Number of epochs without improvement before stopping.
        min_delta: Minimum change in accuracy to count as an improvement.
    """

    def __init__(self, patience: int = 3, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_acc   = -1.0
        self.best_state = None
        self.counter    = 0
        self.stopped    = False

    def step(self, val_acc: float, model: nn.Module) -> bool:
        """Call once per epoch. Returns True if training should stop."""
        if val_acc > self.best_acc + self.min_delta:
            self.best_acc   = val_acc
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True

        return self.stopped

    def restore_best(self, model: nn.Module):
        """Load the best recorded weights back into the model."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def train_one_epoch(epoch, model, train_loader, criterion, optimizer, scaler, scheduler):
    """Train for one epoch and return (avg_loss, train_accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total   = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
    for batch in loop:
        if batch is None:
            continue
        images, labels, _path = batch
        images = images.to(DEVICE, non_blocking=True)
        labels_f = labels.float().unsqueeze(1).to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=DEVICE, dtype=torch.float16, enabled=(DEVICE == "cuda")):
            logits = model(images)
            loss   = criterion(logits, labels_f)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item()

        # Accuracy on training batch (no extra forward pass needed)
        with torch.no_grad():
            preds = (torch.sigmoid(logits).squeeze(1) >= 0.5).long().cpu()
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    avg_loss   = running_loss / max(total // BATCH_SIZE, 1)
    train_acc  = correct / max(total, 1)
    return avg_loss, train_acc


def evaluate(model, loader, split_name: str = "Val"):
    """Evaluate model on a DataLoader. Returns (accuracy, precision, recall, f1)."""
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        with autocast(device_type=DEVICE, dtype=torch.float16, enabled=(DEVICE == "cuda")):
            for batch in loader:
                if batch is None:
                    continue
                images, labels, _path = batch
                images = images.to(DEVICE, non_blocking=True)
                logits = model(images)
                probs  = torch.sigmoid(logits)

                predictions = (probs.float().cpu().numpy() > 0.5).astype(int)
                preds.extend(predictions.flatten())
                trues.extend(labels.numpy().flatten())

    if len(preds) == 0:
        print(f"\n--- {split_name.upper()} RESULTS ---")
        print("No valid samples to evaluate.")
        return 0.0, 0.0, 0.0, 0.0

    accuracy  = float(np.mean(np.array(preds) == np.array(trues)))
    precision = precision_score(trues, preds, zero_division=0)
    recall    = recall_score(trues, preds, zero_division=0)
    f1        = f1_score(trues, preds, zero_division=0)

    print(f"\n--- {split_name.upper()} RESULTS ---")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    return accuracy, precision, recall, f1


if __name__ == "__main__":

    CONV_FILTERS  = [32, 64, 128]          # {16, 32, 64, 128}
    CONV_LAYERS   = [3]            # {1, 2, 3}
    DENSE_NEURONS = [64,128,512,4096]        # {32, 64, 128, 256, 512, 1024, 2048, 4096}
    DENSE_LAYERS  = [1, 2, 3]            # {1, 2, 3}
    KERNEL_SIZES  = [11, 15, 19]            # {1, 2, 3}

    WEIGHT_DECAYS  = [0.0, 1e-4, 1e-3]   # L2 penalty on weights (Adam weight_decay)
    DROPOUT_RATES  = [0.0]#TODO, 0.3, 0.5]     # Dropout probability before each dense layer

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")

    train_loader, val_loader, test_dataset, idx_to_class = data_loaders(DEVICE, BATCH_SIZE)

    #already_done = 8 # counter to skip the models that I already trained

    for CONV_FILTER in CONV_FILTERS:
        for CONV_LAYER in CONV_LAYERS:
            for DENSE_NEURON in DENSE_NEURONS:
                for DENSE_LAYER in DENSE_LAYERS:
                    for WEIGHT_DECAY in WEIGHT_DECAYS:
                        for DROPOUT_RATE in DROPOUT_RATES:
                            for KERNEL_SIZE in KERNEL_SIZES:
                                #if already_done > 0:
                                #    already_done -= 1
                                #    break
                                run_tag = (
                                    f"k[5,9,{KERNEL_SIZE}]_cf{CONV_FILTER}_cl{CONV_LAYER}"
                                    f"_dn{DENSE_NEURON}_dl{DENSE_LAYER}"
                                    f"_wd{WEIGHT_DECAY}_do{DROPOUT_RATE}"
                                )
                                try:
                                    print("-" * 65)
                                    print(
                                        f"STARTING: {run_tag}\n"
                                        f"  kernel_size=[5,9,{KERNEL_SIZE}]\n"
                                        f"  conv_filters={CONV_FILTER}  conv_layers={CONV_LAYER}\n"
                                        f"  dense_neurons={DENSE_NEURON}  dense_layers={DENSE_LAYER}\n"
                                        f"  weight_decay={WEIGHT_DECAY}  dropout={DROPOUT_RATE}"
                                    )

                                    model = I_HAVE_A_THEORY(
                                        KERNEL_SIZE, CONV_FILTER, CONV_LAYER,
                                        DENSE_NEURON, DENSE_LAYER,
                                        dropout_rate=DROPOUT_RATE,
                                    ).to(DEVICE)

                                    criterion = nn.BCEWithLogitsLoss()
                                    optimizer = optim.Adam(
                                        model.parameters(),
                                        lr=LR,
                                        weight_decay=WEIGHT_DECAY,   # L2 regularization
                                    )
                                    scaler = GradScaler(enabled=(DEVICE == "cuda"))
                                    scheduler = optim.lr_scheduler.OneCycleLR(
                                        optimizer,
                                        max_lr=LR,
                                        steps_per_epoch=len(train_loader),
                                        epochs=EPOCHS,
                                        pct_start=0.1,
                                        anneal_strategy="cos",
                                        final_div_factor=1000,
                                    )

                                    early_stop = EarlyStopping(patience=EARLY_STOP)

                                    for epoch in range(EPOCHS):
                                        avg_loss, train_acc = train_one_epoch(
                                            epoch, model, train_loader,
                                            criterion, optimizer, scaler, scheduler,
                                        )

                                        val_acc, _, _, _ = evaluate(model, val_loader, split_name="Val")

                                        gap = train_acc - val_acc
                                        overfit_flag = "Overfit!" if gap > 0.10 else ""
                                        print(
                                            f"  Epoch {epoch+1:>3}/{EPOCHS} | "
                                            f"loss={avg_loss:.4f} | "
                                            f"train_acc={train_acc:.4f} | "
                                            f"val_acc={val_acc:.4f} | "
                                            f"gap={gap:+.4f}{overfit_flag}"
                                        )

                                        ckpt_path = (
                                            f"./models/model_kernel=[5,9,{KERNEL_SIZE}]"
                                            f"_{CONV_FILTER}_{CONV_LAYER}"
                                            f"_{DENSE_NEURON}_{DENSE_LAYER}"
                                            f"_wd{WEIGHT_DECAY}_do{DROPOUT_RATE}.pth"
                                        )
                                        torch.save(model.state_dict(), ckpt_path)

                                        if early_stop.step(val_acc, model):
                                            print(
                                                f"  Early stopping at epoch {epoch+1} "
                                                f"(best val_acc={early_stop.best_acc:.4f}, "
                                                f"no improvement for {EARLY_STOP} epochs)"
                                            )
                                            break

                                    early_stop.restore_best(model)
                                    best_path = (
                                        f"./models/model_kernel=[5,9,{KERNEL_SIZE}]"
                                        f"_{CONV_FILTER}_{CONV_LAYER}"
                                        f"_{DENSE_NEURON}_{DENSE_LAYER}"
                                        f"_wd{WEIGHT_DECAY}_do{DROPOUT_RATE}.pth"
                                    )
                                    torch.save(model.state_dict(), best_path)
                                    print(f"Best checkpoint saved -> {best_path}")

                                    accuracy, precision, recall, f1 = evaluate(
                                        model, val_loader, split_name="Final Val"
                                    )

                                except Exception as e:
                                    print(
                                        f"FAILED: {run_tag}\n"
                                        f"  Error: {e}"
                                    )