# train_deadline.py
# Authors: Team Celestial Blue
# Spring 2025
# Purpose: Train and evaluate satellite turbulence models using k-fold cross-validation.
#          Supports both ConvLSTM and Conv3D architectures.
#          Includes checkpoint/resume for HPC preemption handling.
# Usage: python train_and_test_model.py <seed> <model_type> [--max-samples N] [--data-dir PATH]
#        model_type: "convlstm" or "conv3d"
#        --max-samples: use only the first N .npz files (sorted paths) for a quick smoke test

import argparse
import os
import sys
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

NUM_EPOCHS = 5
NUM_FOLDS = 3
BATCH_SIZE = 16
CROP_SIZE = 128
NUM_BANDS = 6
NUM_FRAMES = 15

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_INPUTS_DIR = os.path.join(SCRIPT_DIR, "..", "model_inputs")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "trained_model_outputs")


# ============================================================
# Model 1: ConvLSTM
# ============================================================
class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding,
        )

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.gates(combined)
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.cell = ConvLSTMCell(in_channels, hidden_channels, kernel_size)

    def forward(self, x):
        batch, seq_len, _, H, W = x.shape
        device = x.device
        h = torch.zeros(batch, self.hidden_channels, H, W, device=device)
        c = torch.zeros(batch, self.hidden_channels, H, W, device=device)
        outputs = []
        for t in range(seq_len):
            h, c = self.cell(x[:, t], h, c)
            outputs.append(h.unsqueeze(1))
        return torch.cat(outputs, dim=1)


class ConvLSTMModel(nn.Module):
    def __init__(self, num_filters=16, dropout=0.2, num_classes=2):
        super().__init__()
        self.convlstm = ConvLSTM(NUM_BANDS, num_filters, kernel_size=3)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(num_filters, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        out = self.convlstm(x)
        out = self.dropout(out)
        out = out.permute(0, 2, 1, 3, 4)
        out = self.classifier(out)
        return out


# ============================================================
# Model 2: 3D CNN
# ============================================================
class Conv3DModel(nn.Module):
    def __init__(self, num_classes=2, dropout=0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(NUM_BANDS, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        out = self.features(x)
        out = self.classifier(out)
        return out


# ============================================================
# Checkpointing for HPC preemption recovery
# ============================================================
MODELS = {
    "convlstm": ConvLSTMModel,
    "conv3d": Conv3DModel,
}


def get_checkpoint_path(model_type, seed, tag=None):
    """tag: e.g. 'max64' when using --max-samples so full runs don't share checkpoints."""
    if tag:
        return os.path.join(OUTPUT_DIR, f"{model_type}_{seed}_{tag}_checkpoint.pth")
    return os.path.join(OUTPUT_DIR, f"{model_type}_{seed}_checkpoint.pth")


def save_checkpoint(path, model, optimizer, l2_alpha_idx, fold, epoch,
                    l2_loss_list, fold_losses, epoch_losses):
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'l2_alpha_idx': l2_alpha_idx,
        'fold': fold,
        'epoch': epoch,
        'l2_loss_list': l2_loss_list,
        'fold_losses': fold_losses,
        'epoch_losses': epoch_losses,
    }
    torch.save(state, path)
    print(f"Checkpoint saved: l2_idx={l2_alpha_idx}, fold={fold}, epoch={epoch}")


def load_checkpoint(path, model, optimizer):
    if os.path.exists(path):
        checkpoint = torch.load(path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded: l2_idx={checkpoint['l2_alpha_idx']}, "
              f"fold={checkpoint['fold']}, epoch={checkpoint['epoch']}")
        return checkpoint
    return None


# ============================================================
# Training and Evaluation
# ============================================================
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for x, y, w in loader:
        x, y, w = x.to(device), y.long().to(device), w.to(device)
        optimizer.zero_grad()
        y_hat = model(x)
        loss = F.cross_entropy(y_hat, y, reduction='none')
        loss = (loss * w).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y, w in loader:
            x, y, w = x.to(device), y.long().to(device), w.to(device)
            y_hat = model(x)
            loss = F.cross_entropy(y_hat, y, reduction='none')
            loss = (loss * w).mean()
            total_loss += loss.item()
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser(
        description="Train/evaluate satellite turbulence models (ConvLSTM or Conv3D)."
    )
    parser.add_argument("seed", type=int)
    parser.add_argument("model_type", type=str)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Use only the first N samples (sorted .npz paths) for a quick smoke test.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing model_inputs tree (default: ../model_inputs under src/).",
    )
    args = parser.parse_args()

    SEED = args.seed
    model_type = args.model_type.lower()

    if model_type not in MODELS:
        print(f"Unknown model type: {model_type}. Choose from: {', '.join(MODELS.keys())}")
        sys.exit(1)

    model_inputs_dir = (
        os.path.abspath(args.data_dir)
        if args.data_dir
        else os.path.abspath(MODEL_INPUTS_DIR)
    )
    smoke_tag = f"max{args.max_samples}" if args.max_samples else None

    ModelClass = MODELS[model_type]
    torch.manual_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"Using device: {device} ({num_gpus} GPU(s))")
    
    if num_gpus > 0 and device.type != "cuda":
        print("WARNING: GPU allocated but PyTorch is using CPU")
    
    print(f"Model type: {model_type}")
    print(f"Data directory: {model_inputs_dir}")

    # Load dataset (lazy loading from .npz files)
    from dataloader_class import SatelliteDataLoader
    dataset = SatelliteDataLoader(model_inputs_dir)
    if args.max_samples is not None:
        n = min(len(dataset), args.max_samples)
        if n < NUM_FOLDS:
            print(
                f"Warning: --max-samples yields n={n} but KFold uses n_splits={NUM_FOLDS}; "
                f"use at least {NUM_FOLDS} samples.",
                file=sys.stderr,
            )
        dataset = Subset(dataset, list(range(n)))
        print(f"Subset smoke test: using {len(dataset)} samples (max {args.max_samples})")
    print(f"Dataset size: {len(dataset)}")

    # Split 85% train, 15% test
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [0.85, 0.15], generator=torch.Generator().manual_seed(SEED)
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    # K-fold CV to find best L2 regularization
    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    l2_values = [0.01, 0]
    checkpoint_path = get_checkpoint_path(model_type, SEED, smoke_tag)

    # Try to resume from checkpoint
    start_l2_idx = 0
    start_fold = 0
    start_epoch = 0
    l2_loss_list = []
    fold_losses = []
    epoch_losses = []
    resuming = False

    # Create a dummy model/optimizer to attempt checkpoint load
    dummy_model = ModelClass().to(device)
    dummy_optimizer = optim.Adam(dummy_model.parameters())
    checkpoint = load_checkpoint(checkpoint_path, dummy_model, dummy_optimizer)

    if checkpoint is not None:
        resuming = True
        start_l2_idx = checkpoint['l2_alpha_idx']
        start_fold = checkpoint['fold']
        start_epoch = checkpoint['epoch']
        l2_loss_list = checkpoint['l2_loss_list']
        fold_losses = checkpoint['fold_losses']
        epoch_losses = checkpoint['epoch_losses']

    for l2_idx in range(start_l2_idx, len(l2_values)):
        l2_alpha = l2_values[l2_idx]
        if l2_idx > start_l2_idx:
            fold_losses = []
        print(f"\nL2 alpha = {l2_alpha}")

        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
            if l2_idx == start_l2_idx and fold < start_fold:
                continue
            if fold > start_fold or l2_idx > start_l2_idx:
                epoch_losses = []

            print(f"  Fold {fold}")
            train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(Subset(train_dataset, val_idx), batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

            if resuming and l2_idx == start_l2_idx and fold == start_fold:
                model = dummy_model
                optimizer = dummy_optimizer
                optimizer.param_groups[0]['weight_decay'] = l2_alpha
                resuming = False
            else:
                model = ModelClass().to(device)
                if num_gpus > 1:
                    model = nn.DataParallel(model)
                optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=l2_alpha)

            first_epoch = start_epoch + 1 if (l2_idx == start_l2_idx and fold == start_fold and start_epoch > 0) else 0
            for epoch in range(first_epoch, NUM_EPOCHS):
                train_loss = train_epoch(model, train_loader, optimizer, device)
                val_loss = evaluate(model, val_loader, device)
                epoch_losses.append(val_loss)
                print(f"    Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
                save_checkpoint(checkpoint_path, model, optimizer, l2_idx, fold, epoch,
                                l2_loss_list, fold_losses, epoch_losses)

            fold_losses.append(epoch_losses[-1])
            print(f"  Fold {fold} final val_loss: {fold_losses[-1]:.4f}")

        # Reset start positions after first l2 pass
        start_fold = 0
        start_epoch = 0

        avg_loss = np.mean(fold_losses)
        l2_loss_list.append(avg_loss)
        print(f"  Average CV loss for l2={l2_alpha}: {avg_loss:.4f}")

    best_l2_idx = np.argmin(l2_loss_list)
    best_l2 = l2_values[best_l2_idx]
    print(f"\nBest L2 alpha: {best_l2} (loss: {l2_loss_list[best_l2_idx]:.4f})")

    # Retrain on full training set with best L2
    print("Retraining on full training set...")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    best_model = ModelClass().to(device)
    if num_gpus > 1:
        best_model = nn.DataParallel(best_model)
    optimizer = optim.Adam(best_model.parameters(), lr=1e-3, weight_decay=best_l2)

    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(best_model, train_loader, optimizer, device)
        print(f"  Retrain epoch {epoch}: loss={train_loss:.4f}")

    # Test evaluation
    print("\nEvaluating on test set...")
    num_correct = 0
    num_true_pos = 0
    num_false_pos = 0
    num_true_neg = 0
    num_false_neg = 0
    total = 0

    best_model.eval()
    with torch.no_grad():
        for x_test, y_test, w_test in test_loader:
            x_test, y_test = x_test.to(device), y_test.long().to(device)
            y_hat = best_model(x_test)
            preds = torch.argmax(y_hat, dim=1)
            for i in range(len(y_test)):
                true_label = y_test[i].item()
                pred_label = preds[i].item()
                total += 1
                if true_label == pred_label:
                    num_correct += 1
                if true_label == 1 and pred_label == 1:
                    num_true_pos += 1
                elif true_label == 0 and pred_label == 1:
                    num_false_pos += 1
                elif true_label == 0 and pred_label == 0:
                    num_true_neg += 1
                elif true_label == 1 and pred_label == 0:
                    num_false_neg += 1

    precision = num_true_pos / max(1, num_true_pos + num_false_pos)
    recall = num_true_pos / max(1, num_true_pos + num_false_neg)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)

    print(f"Accuracy: {num_correct}/{total} ({100 * num_correct / total:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion matrix: TP={num_true_pos}, FP={num_false_pos}, TN={num_true_neg}, FN={num_false_neg}")

    # Save model and clean up checkpoint
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    model_path = os.path.join(OUTPUT_DIR, f"{timestamp}_{model_type}_seed_{SEED}.pth")
    torch.save(best_model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Checkpoint removed (training complete)")


if __name__ == "__main__":
    main()
