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
from ihc_binary import IHCBinaryPatchDataset, iter_markers, load_marker_records  # noqa: E402


try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - depends on sklearn version
    StratifiedGroupKFold = None


BINARY_RESULTS = OUTPUTS_DIR / "binary_ihc_results.txt"
BINARY_METRICS_CSV = OUTPUTS_DIR / "binary_ihc_metrics.csv"
SPLITS_DIR = OUTPUTS_DIR / "binary_ihc_splits"
LOGS_DIR = OUTPUTS_DIR / "logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patient-grouped binary IHC patch classification")
    parser.add_argument("--markers", nargs="+", default=["Her2"], help="Markers: Her2 ER PR Ki67 or all")
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default="convnext")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--limit-per-marker", type=int, default=None, help="Debug only: limit records per marker")
    parser.add_argument("--dry-run", action="store_true", help="Validate data and folds without training")
    return parser.parse_args()


def setup_logger(marker: str, args: argparse.Namespace) -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"binary_{marker}_{args.model}_seed_{args.seed}_{timestamp}.log"

    logger = logging.getLogger(f"binary_ihc_{marker}")
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


def make_splitter(args: argparse.Namespace):
    if StratifiedGroupKFold is not None:
        return StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed), "StratifiedGroupKFold"
    return GroupKFold(n_splits=args.folds), "GroupKFold"


def get_splits(labels: np.ndarray, groups: np.ndarray, args: argparse.Namespace):
    splitter, splitter_name = make_splitter(args)
    indices = np.arange(len(labels))
    if splitter_name == "StratifiedGroupKFold":
        return list(splitter.split(indices, labels, groups)), splitter_name
    return list(splitter.split(indices, labels, groups)), splitter_name


def class_weight_tensor(labels: list[int], device: torch.device):
    counts = Counter(labels)
    total = len(labels)
    weights = []
    for class_idx in [0, 1]:
        count = counts.get(class_idx, 0)
        weights.append(total / (2 * count) if count else 0.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, epochs, amp_enabled, scaler, channels_last):
    model.train()
    running_loss = 0.0

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

        running_loss += loss.item()

    return running_loss / max(len(loader), 1)


def evaluate(model, loader, criterion, device, amp_enabled, channels_last):
    model.eval()
    running_loss = 0.0
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

            running_loss += loss.item()
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_score.extend(probs[:, 1].detach().cpu().numpy().tolist())

    metrics = compute_metrics(y_true, y_pred, y_score)
    metrics["loss"] = running_loss / max(len(loader), 1)
    return metrics


def compute_metrics(y_true, y_pred, y_score):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "roc_auc": auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def append_metrics_csv(rows: list[dict]) -> None:
    BINARY_METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    csv_exists = BINARY_METRICS_CSV.exists()
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
    with BINARY_METRICS_CSV.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not csv_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_split_csv(marker: str, fold: int, records, train_idx, val_idx):
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    split_path = SPLITS_DIR / f"{marker}_fold_{fold}.csv"
    with split_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["split", "patient_id", "label", "path"])
        for split_name, indices in (("train", train_idx), ("val", val_idx)):
            for idx in indices:
                record = records[idx]
                writer.writerow([split_name, record.patient_id, record.label, record.path])


def summarize_records(marker: str, records, logger: logging.Logger):
    label_counts = Counter(record.label for record in records)
    patients = sorted({record.patient_id for record in records})
    logger.info("Marker: %s", marker)
    logger.info("Images: %s", len(records))
    logger.info("Patients: %s", len(patients))
    logger.info("Label counts: %s", dict(label_counts))
    logger.info("Class meaning: 0=negative, 1=positive")


def train_marker(marker: str, args: argparse.Namespace):
    logger = setup_logger(marker, args)
    records = load_marker_records(marker)
    if args.limit_per_marker is not None:
        records = records[: args.limit_per_marker]

    summarize_records(marker, records, logger)

    labels = np.array([record.label for record in records])
    groups = np.array([record.patient_id for record in records])
    splits, splitter_name = get_splits(labels, groups, args)
    logger.info("Splitter: %s", splitter_name)

    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        train_patients = {records[idx].patient_id for idx in train_idx}
        val_patients = {records[idx].patient_id for idx in val_idx}
        overlap = train_patients.intersection(val_patients)
        train_counts = Counter(records[idx].label for idx in train_idx)
        val_counts = Counter(records[idx].label for idx in val_idx)
        logger.info(
            "Fold %s/%s | train=%s imgs/%s patients %s | val=%s imgs/%s patients %s | overlap=%s",
            fold,
            args.folds,
            len(train_idx),
            len(train_patients),
            dict(train_counts),
            len(val_idx),
            len(val_patients),
            dict(val_counts),
            len(overlap),
        )
        write_split_csv(marker, fold, records, train_idx, val_idx)

    if args.dry_run:
        logger.info("Dry run requested; skipping training.")
        return []

    device = get_device()
    configure_torch_performance(device, deterministic=args.deterministic)
    pin_memory = device.type == "cuda"
    amp_enabled = device.type == "cuda" and not args.no_amp
    use_augmentation = not args.no_augmentation
    train_transform, val_transform = get_transforms(use_augmentation=use_augmentation)

    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("CUDA device: %s", torch.cuda.get_device_name(0))
    logger.info("AMP enabled: %s", amp_enabled)
    logger.info("Pretrained: %s", not args.no_pretrained)

    fold_rows = []
    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        logger.info("=== Training fold %s/%s ===", fold, args.folds)
        train_records = [records[idx] for idx in train_idx]
        val_records = [records[idx] for idx in val_idx]
        train_dataset = IHCBinaryPatchDataset(train_records, transform=train_transform)
        val_dataset = IHCBinaryPatchDataset(val_records, transform=val_transform)

        loader_kwargs = {
            "num_workers": args.num_workers,
            "pin_memory": pin_memory,
        }
        if args.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = args.prefetch_factor

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            **loader_kwargs,
        )

        model = get_model(args.model, num_classes=2, pretrained=not args.no_pretrained).to(device)
        if args.channels_last and device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)

        weights = None if args.no_class_weights else class_weight_tensor([r.label for r in train_records], device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

        best_val_loss = float("inf")
        best_metrics = None
        patience_counter = 0
        best_path = MODELS_DIR / f"binary_{marker}_{args.model}_fold_{fold}_seed_{args.seed}_best.pth"
        last_path = MODELS_DIR / f"binary_{marker}_{args.model}_fold_{fold}_seed_{args.seed}_last.pth"

        for epoch in range(args.epochs):
            train_loss = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                epoch,
                args.epochs,
                amp_enabled,
                scaler,
                args.channels_last and device.type == "cuda",
            )
            metrics = evaluate(
                model,
                val_loader,
                criterion,
                device,
                amp_enabled,
                args.channels_last and device.type == "cuda",
            )

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
                "num_classes": 2,
                "class_names": ["negative", "positive"],
                "seed": args.seed,
                "fold": fold,
                "image_size": 224,
            }
            torch.save(checkpoint, last_path)

            if metrics["loss"] < best_val_loss:
                best_val_loss = metrics["loss"]
                best_metrics = metrics
                patience_counter = 0
                torch.save(checkpoint, best_path)
                logger.info("New best checkpoint: %s", best_path)
            else:
                patience_counter += 1
                logger.info("No val improvement for %s epoch(s)", patience_counter)

            if patience_counter >= args.patience:
                logger.info("Early stopping triggered")
                break

        if best_metrics is None:
            raise RuntimeError(f"Fold {fold} finished without validation metrics")

        train_patients = {record.patient_id for record in train_records}
        val_patients = {record.patient_id for record in val_records}
        row = {
            "marker": marker,
            "model": args.model,
            "seed": args.seed,
            "fold": fold,
            "splitter": splitter_name,
            "train_images": len(train_records),
            "val_images": len(val_records),
            "train_patients": len(train_patients),
            "val_patients": len(val_patients),
            **best_metrics,
        }
        fold_rows.append(row)

    append_metrics_csv(fold_rows)
    write_summary(marker, args, fold_rows, splitter_name, logger)
    return fold_rows


def metric_mean_std(rows, key: str):
    values = np.array([row[key] for row in rows], dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(values.mean()), float(values.std(ddof=1))


def write_summary(marker: str, args: argparse.Namespace, rows: list[dict], splitter_name: str, logger: logging.Logger):
    lines = [
        f"Binary IHC CV Summary ({marker}, {args.model}, {args.folds}-fold, seed {args.seed})",
        f"Splitter: {splitter_name}",
    ]
    for key in ["accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc"]:
        mean, std = metric_mean_std(rows, key)
        text = "N/A" if np.isnan(mean) else f"{mean:.4f} +/- {std:.4f}"
        lines.append(f"{key}: {text}")
    lines.append("")
    summary = "\n".join(lines)

    with BINARY_RESULTS.open("a", encoding="utf-8") as file:
        file.write(summary + "\n")
    logger.info("\n%s", summary)


def main() -> None:
    args = parse_args()
    set_seed(args.seed, deterministic=args.deterministic)
    ensure_project_dirs()

    requested_markers = []
    for marker_arg in args.markers:
        requested_markers.extend(iter_markers(marker_arg))
    markers = list(dict.fromkeys(requested_markers))

    all_rows = []
    for marker in markers:
        all_rows.extend(train_marker(marker, args))

    if args.dry_run:
        print("Dry run complete. Split files were written to:", SPLITS_DIR)
    elif all_rows:
        print("Training complete. Metrics written to:", BINARY_METRICS_CSV)


if __name__ == "__main__":
    main()
