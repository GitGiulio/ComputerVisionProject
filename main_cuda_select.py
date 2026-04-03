import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
#from tqdm import tqdm
from torch.amp import autocast, GradScaler
from shared_code import SafeImageFolder, collate_skip_none, data_loaders, CIFAKE_CNN,I_HAVE_A_THEORY, parse_hparams_from_model_path, GradCAM


device_number=1

DATA_DIR = "/mnt/scratch/Stable_diffusion/Stable_diffusion_ready"
BATCH_SIZE = 256
LR = 1e-3
EPOCHS = 20

torch.backends.cudnn.benchmark = True

# Model hyperparameters (from paper search space)
CONV_FILTER = None          # {16, 32, 64, 128}
CONV_LAYER = None            # {1, 2, 3}
DENSE_NEURON = None         # {32, 64, 128, 256, 512, 1024, 2048, 4096}
DENSE_LAYER = None           # {1, 2, 3}

# Grad-CAM / output config
OUT_DIR = "/home/cv04f26/ComputerVisionProject/gradcam_outputs"
NUM_CAM_SAMPLES = 12
IMAGE_SIZE = 256


def train_one_epoch(epoch,train_loader):
    model.train()
    running_loss = 0.0
    steps = 0

    #loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
    for i,batch in enumerate(train_loader):
        #print(f"done batch {i}")
        if batch is None:
            continue
        images, labels, _path = batch
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=DEVICE, dtype=torch.float16, enabled=(DEVICE == f"cuda:{device_number}")):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        steps += 1
        #loop.set_postfix(loss=(running_loss / max(steps, 1)))

    avg_loss = running_loss / max(steps, 1)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

def evaluate(model,test_loader):
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        with autocast(device_type=DEVICE, dtype=torch.float16, enabled=(DEVICE == f"cuda:{device_number}")):
            for batch in test_loader:
                if batch is None:
                    continue
                images, labels, _path = batch
                images = images.to(DEVICE, non_blocking=True)
                logits = model(images)
                probs = torch.sigmoid(logits)

                predictions = (probs.float().cpu().numpy() > 0.5).astype(int)
                preds.extend(predictions.flatten())
                trues.extend(labels.numpy().flatten())

    if len(preds) == 0:
        print("\n--- TEST RESULTS ---")
        print("No valid samples to evaluate.")
        return 0, 0, 0, 0

    accuracy  = np.mean(np.array(preds) == np.array(trues))
    precision = precision_score(trues, preds, zero_division=0)
    recall    = recall_score(trues, preds, zero_division=0)
    f1        = f1_score(trues, preds, zero_division=0)

    print("\n--- TEST RESULTS ---")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    return accuracy, precision, recall, f1


if __name__ == "__main__":
    KERNEL_SIZES = [3,11,21]
    CONV_FILTERS = [32,64]          # {16, 32, 64, 128}
    CONV_LAYERS = [2, 3]            # {1, 2, 3}
    DENSE_NEURONS = [64,4096]         # {32, 64, 128, 256, 512, 1024, 2048, 4096}
    DENSE_LAYERS = [1,3]           # {1, 2, 3}
    
    ok = True
    while ok:
        device_number = 0
        #int(input("WHAT GPU DO YOU WANNA USE? 0 or 1?"))
        if device_number in [0,1]:
            ok = False
    DEVICE = f"cuda:{device_number}" if torch.cuda.is_available() else "cpu"

    print(DEVICE)
    train_loader, test_loader, idx_to_class = data_loaders(DEVICE,BATCH_SIZE)
    CONV_FILTER = CONV_FILTERS[device_number]
    for CONV_LAYER in CONV_LAYERS:
        for DENSE_NEURON in DENSE_NEURONS:
            for DENSE_LAYER in DENSE_LAYERS:
                KERNEL_SIZE = KERNEL_SIZES[2]
                try:
                    print("---------------------------------------------------------")
                    print(f"STARTING TRAINING with \n CONV_FILTERS:{CONV_FILTER}\n CONV_LAYERS:{CONV_LAYER}\n DENSE_NEURONS:{DENSE_NEURON}\n DENSE_LAYERS:{DENSE_LAYER}")
                    #model = CIFAKE_CNN(CONV_FILTER, CONV_LAYER, DENSE_NEURON, DENSE_LAYER).to(DEVICE)
                    model = I_HAVE_A_THEORY(KERNEL_SIZE,CONV_FILTER, CONV_LAYER, DENSE_NEURON, DENSE_LAYER).to(DEVICE)
                    print(model)
                    # Use logits-safe, autocast-safe BCEWithLogitsLoss
                    criterion = nn.BCEWithLogitsLoss()
                    optimizer = optim.Adam(model.parameters(), lr=LR)
                    scaler = GradScaler(enabled=(DEVICE == f"cuda:{device_number}"))
                    for epoch in range(EPOCHS):
                        train_one_epoch(epoch,train_loader)
                        torch.save(obj=model.state_dict(),f=f"/home/cv04f26/ComputerVisionProject/models/model_kernel={KERNEL_SIZE}_{CONV_FILTER}_{CONV_LAYER}_{DENSE_NEURON}_{DENSE_LAYER}.pth")
                    accuracy, precision, recall, f1 = evaluate(model,train_loader)
                except Exception as e:
                    print(f"FAILED with \n CONV_FILTERS:{CONV_FILTER}\n CONV_LAYERS:{CONV_LAYER}\n DENSE_NEURONS:{DENSE_NEURON}\n DENSE_LAYERS:{DENSE_LAYER}")
                    print(e)