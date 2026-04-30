#!/usr/bin/env python3
# check_progress.py
# Authors: Razzle Dazzle Rose
# Spring 2026
# Usage: python check_progress.py <sat_train.[number].out> 
#                                [path/to/train_and_test_model.py]
#
# If the second argument is omitted, looks for train_and_test_model.py at the
# default repo location relative to this script:
#   ../src/train_and_test_model.py
# use to check the progress of a model being trained

import sys
import re
import os


DEFAULT_TRAIN_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../..", "src", "train_and_test_model.py"
)


def load_training_config(train_script_path):
    """Parse NUM_EPOCHS, NUM_FOLDS, and l2_values out of train_and_test_model.py."""
    path = os.path.abspath(train_script_path)
    if not os.path.exists(path):
        print(f"\n  ERROR: Could not find train_and_test_model.py at:\n    {path}")
        print("\n  Pass the correct path as a second argument:")
        print("  python check_progress.py <log_file> <path/to/train_and_test_model.py>\n")
        sys.exit(1)

    num_epochs = None
    num_folds  = None
    l2_values  = None

    with open(path) as f:
        for line in f:
            stripped = line.strip()
            m = re.match(r"^NUM_EPOCHS\s*=\s*(\d+)", stripped)
            if m:
                num_epochs = int(m.group(1))
            m = re.match(r"^NUM_FOLDS\s*=\s*(\d+)", stripped)
            if m:
                num_folds = int(m.group(1))
            m = re.search(r"l2_values\s*=\s*\[([^\]]+)\]", stripped)
            if m:
                l2_values = [float(x.strip()) for x in m.group(1).split(",")]

    missing = [name for name, val in
               [("NUM_EPOCHS", num_epochs), ("NUM_FOLDS", num_folds), ("l2_values", l2_values)]
               if val is None]
    if missing:
        print(f"\n  ERROR: Could not parse {', '.join(missing)} from:\n    {path}\n")
        sys.exit(1)

    return num_epochs, num_folds, l2_values


def parse_log(path, l2_values):
    epochs      = []  # dicts: l2_idx, fold, epoch, train_loss, val_loss
    fold_finals = []  # dicts: l2_idx, fold, val_loss
    l2_avgs     = []  # list of (l2_alpha, avg_loss)

    epoch_re      = re.compile(r"Epoch\s+(\d+):\s+train_loss=([\d.]+),\s+val_loss=([\d.]+)")
    fold_re       = re.compile(r"Fold\s+(\d+)\s+final val_loss:\s+([\d.]+)")
    l2_re         = re.compile(r"L2 alpha\s*=\s*([\d.]+)")
    avg_re        = re.compile(r"Average CV loss for l2=([\d.]+):\s+([\d.]+)")
    fold_start_re = re.compile(r"Fold\s+(\d+)$")

    current_l2_idx = 0
    current_fold   = 0

    with open(path) as f:
        for line in f:
            line = line.strip()

            m = l2_re.search(line)
            if m:
                alpha = float(m.group(1))
                if alpha in l2_values:
                    current_l2_idx = l2_values.index(alpha)
                continue

            m = fold_start_re.search(line)
            if m:
                current_fold = int(m.group(1))
                continue

            m = epoch_re.search(line)
            if m:
                epochs.append({
                    'l2_idx':     current_l2_idx,
                    'fold':       current_fold,
                    'epoch':      int(m.group(1)),
                    'train_loss': float(m.group(2)),
                    'val_loss':   float(m.group(3)),
                })
                continue

            m = fold_re.search(line)
            if m:
                fold_finals.append({
                    'l2_idx':   current_l2_idx,
                    'fold':     current_fold,
                    'val_loss': float(m.group(2)),
                })
                continue

            m = avg_re.search(line)
            if m:
                l2_avgs.append((float(m.group(1)), float(m.group(2))))

    return epochs, fold_finals, l2_avgs


def summarize(log_path, train_script_path):
    num_epochs, num_folds, l2_values = load_training_config(train_script_path)
    total_fold_runs = len(l2_values) * num_folds

    print(f"\n{'='*58}")
    print(f"  TRAINING PROGRESS REPORT")
    print(f"  Log:    {log_path}")
    print(f"  Config: NUM_EPOCHS={num_epochs}, NUM_FOLDS={num_folds}, "
          f"l2_values={l2_values}")
    print(f"{'='*58}")

    try:
        epochs, fold_finals, l2_avgs = parse_log(log_path, l2_values)
    except FileNotFoundError:
        print(f"\n  ERROR: Log file not found: {log_path}\n")
        sys.exit(1)

    if not epochs:
        print("\n  No epoch data found yet — job may still be starting up.\n")
        return

    # ── Overall progress ─────────────────────────────────────────────
    completed_folds       = len(fold_finals)
    last                  = epochs[-1]
    current_l2            = l2_values[last['l2_idx']]
    current_fold          = last['fold']
    current_epoch         = last['epoch']
    epochs_done_this_fold = current_epoch + 1
    total_epochs_done     = completed_folds * num_epochs + epochs_done_this_fold
    total_cv_epochs       = total_fold_runs * num_epochs
    pct                   = 100 * total_epochs_done / total_cv_epochs

    print(f"\n  CURRENT POSITION")
    print(f"  {'L2 alpha:':<22} {current_l2}  (run {last['l2_idx']+1} of {len(l2_values)})")
    print(f"  {'Fold:':<22} {current_fold} of {num_folds-1}")
    print(f"  {'Epoch:':<22} {current_epoch} of {num_epochs-1}")
    print(f"  {'Folds completed:':<22} {completed_folds} of {total_fold_runs}")
    print(f"  {'CV epochs done:':<22} {total_epochs_done} of {total_cv_epochs}")
    print(f"  {'Overall progress:':<22} {pct:.1f}%")
    bar_len = 40
    filled  = int(bar_len * pct / 100)
    bar     = "█" * filled + "░" * (bar_len - filled)
    print(f"\n  [{bar}] {pct:.1f}%\n")

    # ── Current fold loss trend ───────────────────────────────────────
    current_fold_epochs = [
        e for e in epochs
        if e['l2_idx'] == last['l2_idx'] and e['fold'] == last['fold']
    ]
    if len(current_fold_epochs) >= 2:
        first_val = current_fold_epochs[0]['val_loss']
        last_val  = current_fold_epochs[-1]['val_loss']
        delta     = last_val - first_val
        trend     = "↓ improving" if delta < -0.001 else ("↑ worsening" if delta > 0.001 else "→ flat")
        print(f"  CURRENT FOLD LOSS TREND")
        print(f"  {'First epoch val_loss:':<26} {first_val:.4f}")
        print(f"  {'Latest val_loss:':<26} {last_val:.4f}  ({trend})")
        print(f"  {'Change:':<26} {delta:+.4f}")

        if len(current_fold_epochs) >= 5:
            recent = [e['val_loss'] for e in current_fold_epochs[-5:]]
            spread = max(recent) - min(recent)
            if spread < 0.001:
                print(f"\n  ⚠  WARNING: val_loss has been flat for the last 5 epochs "
                      f"(spread={spread:.4f}). Possible plateau.")

    # ── Completed L2 runs ─────────────────────────────────────────────
    if l2_avgs:
        print(f"\n  COMPLETED L2 RUNS")
        best_avg = min(a for _, a in l2_avgs)
        for alpha, avg in l2_avgs:
            marker = " ← best so far" if avg == best_avg else ""
            print(f"  L2={alpha:<8}  avg val_loss={avg:.4f}{marker}")

    # ── Best val_loss seen so far ─────────────────────────────────────
    best = min(epochs, key=lambda e: e['val_loss'])
    print(f"\n  BEST VAL LOSS SO FAR")
    print(f"  {best['val_loss']:.4f}  "
          f"(L2={l2_values[best['l2_idx']]}, fold={best['fold']}, epoch={best['epoch']})")

    print(f"\n{'='*58}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("\nUsage: python check_progress.py <log_file> [path/to/train_and_test_model.py]")
        print("  log_file             — SLURM .out file from your training job")
        print("  train_and_test_model — optional, defaults to ../src/train_and_test_model.py\n")
        sys.exit(1)

    log_path          = sys.argv[1]
    train_script_path = sys.argv[2] if len(sys.argv) == 3 else DEFAULT_TRAIN_SCRIPT
    summarize(log_path, train_script_path)