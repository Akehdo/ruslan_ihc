import argparse
import csv
import logging
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common import (  # noqa: E402
    MODELS_DIR,
    OUTPUTS_DIR,
    SUPPORTED_MODELS,
    configure_torch_performance,
    ensure_project_dirs,
    get_device,
    get_model,
    get_transforms,
    set_seed,
)
from ihc_binary import IHCBinaryPatchDataset, iter_markers, load_manifest_records  # noqa: E402


try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover
    StratifiedGroupKFold = None


RESULTS_TXT = OUTPUTS_DIR / "ihc_binary_results.txt"
METRICS_CSV = OUTPUTS_DIR / "ihc_binary_metrics.csv"
SPLITS_DIR = OUTPUTS_DIR / "ihc_binary_splits"
LOGS_DIR = OUTPUTS_DIR / "logs"


def parse_args():
    parser = argparse.ArgumentParser(description="Train binary IHC positive/negative classifiers")
    parser.add_argument("--manifest", default="outputs/ihc_binary_dataset/manifest.csv")
    parser.add_argument("--markers", nargs="+", default=["all"], help="Markers: Her2 ER PR Ki67 or all")
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default="convnext")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Create patient folds without training")
    return parser.parse_args()


def setup_logger(marker: str, args) -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"ihc_binary_{marker}_{args.model}_seed_{args.seed}_{timestamp}.log"

    logger = logging.getLogger(f"ihc_binary_{marker}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Log file: %s", log_path)
    return logger


def make_splits(labels: np.ndarray, groups: np.ndarray, folds: int, seed: int):
    indices = np.arange(len(labels))
    if StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        return list(splitter.split(indices, labels, groups)), "StratifiedGroupKFold"

    splitter = GroupKFold(n_splits=folds)
    return list(splitter.split(indices, labels, groups)), "GroupKFold"


def class_weights(labels: list[int], device: torch.device):
    counts = Counter(labels)
    total = len(labels)
    values = [total / (2 * counts[label]) if counts[label] else 0.0 for label in [0, 1]]
    return torch.tensor(values, dtype=torch.float32, device=device)


def train_epoch(model, loader, criterion, optimizer, device, epoch, epochs, amp_enabled, scaler, channels_last):
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", file=sys.stdout):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def evaluate(model, loader, criterion, device, amp_enabled, channels_last):
    model.eval()
    total_loss = 0.0
    y_true = []
    y_pred = []
    y_score = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if channels_last:
                images = images.contiguous(memory_format=torch.channels_last)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            total_loss += loss.item()
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_score.extend(probs[:, 1].cpu().tolist())

    metrics = compute_metrics(y_true, y_pred, y_score)
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


def compute_metrics(y_true, y_pred, y_score):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except ValueError:
        roc_auc = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "roc_auc": roc_auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def write_split(marker: str, fold: int, records, train_idx, val_idx):
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    split_path = SPLITS_DIR / f"{marker}_fold_{fold}.csv"
    with split_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["split", "patient_id", "label", "path"])
        for split_name, indices in [("train", train_idx), ("val", val_idx)]:
            for idx in indices:
                record = records[idx]
                writer.writerow([split_name, record.patient_id, record.label, record.path])


def append_metrics(rows: list[dict]):
    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    exists = METRICS_CSV.exists()
    fieldnames = [
        "marker",
        "model",
        "seed",
        "fold",
        "splitter",
        "train_images",
        "val_images",
        "train_patients",
        "val_patients",
        "loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "roc_auc",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    with METRICS_CSV.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def mean_std(rows: list[dict], key: str):
    values = np.array([row[key] for row in rows], dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return "N/A"
    if len(values) == 1:
        return f"{values[0]:.4f} +/- 0.0000"
    return f"{values.mean():.4f} +/- {values.std(ddof=1):.4f}"


def write_summary(marker: str, args, rows: list[dict], splitter_name: str, logger):
    lines = [
        f"Binary IHC Summary ({marker}, {args.model}, {args.folds}-fold, seed {args.seed})",
        f"Splitter: {splitter_name}",
    ]
    for key in ["accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc"]:
        lines.append(f"{key}: {mean_std(rows, key)}")
    lines.append("")
    text = "\n".join(lines)
    with RESULTS_TXT.open("a", encoding="utf-8") as file:
        file.write(text + "\n")
    logger.info("\n%s", text)


def train_marker(marker: str, args):
    logger = setup_logger(marker, args)
    records = load_manifest_records(Path(args.manifest), marker)
    labels = np.array([record.label for record in records])
    groups = np.array([record.patient_id for record in records])
    splits, splitter_name = make_splits(labels, groups, args.folds, args.seed)

    logger.info("Marker: %s", marker)
    logger.info("Manifest: %s", args.manifest)
    logger.info("Images: %s", len(records))
    logger.info("Patients: %s", len(set(groups)))
    logger.info("Label counts: %s", dict(Counter(labels)))
    logger.info("Class meaning: 0=negative, 1=positive")
    logger.info("Splitter: %s", splitter_name)

    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        train_patients = {records[idx].patient_id for idx in train_idx}
        val_patients = {records[idx].patient_id for idx in val_idx}
        logger.info(
            "Fold %s/%s | train=%s imgs/%s patients %s | val=%s imgs/%s patients %s | overlap=%s",
            fold,
            args.folds,
            len(train_idx),
            len(train_patients),
            dict(Counter(records[idx].label for idx in train_idx)),
            len(val_idx),
            len(val_patients),
            dict(Counter(records[idx].label for idx in val_idx)),
            len(train_patients.intersection(val_patients)),
        )
        write_split(marker, fold, records, train_idx, val_idx)

    if args.dry_run:
        logger.info("Dry run requested; skipping training.")
        return []

    device = get_device()
    configure_torch_performance(device)
    amp_enabled = device.type == "cuda" and not args.no_amp
    channels_last = args.channels_last and device.type == "cuda"
    pin_memory = device.type == "cuda"
    train_transform, val_transform = get_transforms(use_augmentation=not args.no_augmentation)

    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("CUDA device: %s", torch.cuda.get_device_name(0))
    logger.info("AMP enabled: %s", amp_enabled)

    rows = []
    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        logger.info("=== Training fold %s/%s ===", fold, args.folds)
        train_records = [records[idx] for idx in train_idx]
        val_records = [records[idx] for idx in val_idx]
        train_loader = DataLoader(
            IHCBinaryPatchDataset(train_records, train_transform),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            IHCBinaryPatchDataset(val_records, val_transform),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )

        model = get_model(args.model, num_classes=2, pretrained=not args.no_pretrained).to(device)
        if channels_last:
            model = model.to(memory_format=torch.channels_last)

        weights = None if args.no_class_weights else class_weights([r.label for r in train_records], device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

        best_loss = float("inf")
        best_metrics = None
        patience_count = 0
        best_path = MODELS_DIR / f"ihc_binary_{marker}_{args.model}_fold_{fold}_seed_{args.seed}_best.pth"
        last_path = MODELS_DIR / f"ihc_binary_{marker}_{args.model}_fold_{fold}_seed_{args.seed}_last.pth"

        for epoch in range(args.epochs):
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch, args.epochs, amp_enabled, scaler, channels_last
            )
            metrics = evaluate(model, val_loader, criterion, device, amp_enabled, channels_last)
            logger.info(
                "Epoch [%s/%s] train_loss=%.4f val_loss=%.4f acc=%.4f f1=%.4f auc=%s",
                epoch + 1,
                args.epochs,
                train_loss,
                metrics["loss"],
                metrics["accuracy"],
                metrics["f1"],
                f"{metrics['roc_auc']:.4f}" if not np.isnan(metrics["roc_auc"]) else "N/A",
            )

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_name": args.model,
                "marker": marker,
                "class_names": ["negative", "positive"],
                "num_classes": 2,
                "seed": args.seed,
                "fold": fold,
            }
            torch.save(checkpoint, last_path)

            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                best_metrics = metrics
                patience_count = 0
                torch.save(checkpoint, best_path)
                logger.info("New best checkpoint: %s", best_path)
            else:
                patience_count += 1

            if patience_count >= args.patience:
                logger.info("Early stopping triggered")
                break

        row = {
            "marker": marker,
            "model": args.model,
            "seed": args.seed,
            "fold": fold,
            "splitter": splitter_name,
            "train_images": len(train_records),
            "val_images": len(val_records),
            "train_patients": len({r.patient_id for r in train_records}),
            "val_patients": len({r.patient_id for r in val_records}),
            **best_metrics,
        }
        rows.append(row)

    append_metrics(rows)
    write_summary(marker, args, rows, splitter_name, logger)
    return rows


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_project_dirs()

    markers = []
    for marker_arg in args.markers:
        markers.extend(iter_markers(marker_arg))
    markers = list(dict.fromkeys(markers))

    all_rows = []
    for marker in markers:
        all_rows.extend(train_marker(marker, args))

    if args.dry_run:
        print("Dry run complete. Split files:", SPLITS_DIR)
    elif all_rows:
        print("Training complete. Metrics:", METRICS_CSV)


if __name__ == "__main__":
    main()
