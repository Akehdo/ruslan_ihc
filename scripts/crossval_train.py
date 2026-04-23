import argparse
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common import (  # noqa: E402
    MODELS_DIR,
    OUTPUTS_DIR,
    SUPPORTED_MODELS,
    TRAIN_DIR,
    ensure_project_dirs,
    configure_torch_performance,
    get_device,
    get_model,
    get_transforms,
    set_seed,
    validate_dataset_dirs,
)


CV_RESULTS = OUTPUTS_DIR / "cv_results.txt"
CV_METRICS_CSV = OUTPUTS_DIR / "cv_metrics.csv"
LOGS_DIR = OUTPUTS_DIR / "logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-validation training for HER2 classifiers")
    parser.add_argument("--model", choices=SUPPORTED_MODELS, required=True)
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
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    return parser.parse_args()


def setup_logger(args: argparse.Namespace) -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"{args.model}_seed_{args.seed}_{timestamp}.log"

    logger = logging.getLogger("crossval_train")
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


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    epoch,
    epochs,
    amp_enabled,
    scaler,
    channels_last,
) -> float:
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", file=sys.stdout):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


def evaluate_epoch(model, val_loader, criterion, device, amp_enabled, channels_last) -> tuple[float, dict]:
    model.eval()
    running_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if channels_last:
                images = images.contiguous(memory_format=torch.channels_last)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(images)
                loss = criterion(outputs, labels)
            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    val_loss = running_loss / len(val_loader)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return val_loss, report


def train_one_fold(
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    lr,
    patience,
    best_path,
    last_path,
    amp_enabled,
    channels_last,
    logger,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_val_loss = float("inf")
    patience_counter = 0
    best_report = None

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            epochs,
            amp_enabled,
            scaler,
            channels_last,
        )
        val_loss, report = evaluate_epoch(model, val_loader, criterion, device, amp_enabled, channels_last)

        logger.info(
            "Epoch [%s/%s] train_loss=%.4f val_loss=%.4f accuracy=%.4f macro_f1=%.4f",
            epoch + 1,
            epochs,
            train_loss,
            val_loss,
            report["accuracy"],
            report["macro avg"]["f1-score"],
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_report = report
            torch.save(model.state_dict(), best_path)
            logger.info("New best model saved: %s", best_path)
        else:
            patience_counter += 1
            logger.info("No val improvement for %s epoch(s)", patience_counter)

        torch.save(model.state_dict(), last_path)

        if patience_counter >= patience:
            logger.info("Early stopping triggered")
            break

    return best_report


def append_fold_metrics(args: argparse.Namespace, fold_reports: list[dict]) -> None:
    csv_exists = CV_METRICS_CSV.exists()
    with CV_METRICS_CSV.open("a", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "model",
            "seed",
            "fold",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_precision",
            "weighted_recall",
            "weighted_f1",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not csv_exists:
            writer.writeheader()

        for fold_idx, report in enumerate(fold_reports, 1):
            writer.writerow({
                "model": args.model,
                "seed": args.seed,
                "fold": fold_idx,
                "accuracy": report["accuracy"],
                "macro_precision": report["macro avg"]["precision"],
                "macro_recall": report["macro avg"]["recall"],
                "macro_f1": report["macro avg"]["f1-score"],
                "weighted_precision": report["weighted avg"]["precision"],
                "weighted_recall": report["weighted avg"]["recall"],
                "weighted_f1": report["weighted avg"]["f1-score"],
            })


def main() -> None:
    args = parse_args()
    logger = setup_logger(args)
    set_seed(args.seed, deterministic=args.deterministic)
    ensure_project_dirs()
    validate_dataset_dirs()

    device = get_device()
    configure_torch_performance(device, deterministic=args.deterministic)
    pin_memory = device.type == "cuda"
    use_augmentation = not args.no_augmentation
    amp_enabled = device.type == "cuda" and not args.no_amp

    train_transform, val_transform = get_transforms(use_augmentation=use_augmentation)

    full_dataset = ImageFolder(TRAIN_DIR, transform=train_transform)
    targets = full_dataset.targets

    logger.info("Args: %s", vars(args))
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("CUDA device: %s", torch.cuda.get_device_name(0))
    logger.info("Dataset: %s", TRAIN_DIR)
    logger.info("Classes: %s", full_dataset.class_to_idx)
    logger.info("Train images: %s", len(full_dataset))
    logger.info("AMP enabled: %s", amp_enabled)
    logger.info("Channels last: %s", args.channels_last)
    logger.info("TF32 enabled: True")
    logger.info("CuDNN benchmark: %s", torch.backends.cudnn.benchmark)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_reports = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(targets)), targets), 1):
        logger.info("=== Fold %s/%s ===", fold, args.folds)

        train_subset = Subset(full_dataset, train_idx)
        val_dataset = ImageFolder(TRAIN_DIR, transform=val_transform)
        val_subset = Subset(val_dataset, val_idx)

        loader_kwargs = {
            "num_workers": args.num_workers,
            "pin_memory": pin_memory,
        }
        if args.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = args.prefetch_factor

        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            **loader_kwargs,
        )

        model = get_model(args.model).to(device)
        if args.channels_last and device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)

        best_path = MODELS_DIR / f"{args.model}_cv_fold_{fold}_seed_{args.seed}_best.pth"
        last_path = MODELS_DIR / f"{args.model}_cv_fold_{fold}_seed_{args.seed}_last.pth"
        logger.info("Best checkpoint: %s", best_path)
        logger.info("Last checkpoint: %s", last_path)

        report = train_one_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            best_path=best_path,
            last_path=last_path,
            amp_enabled=amp_enabled,
            channels_last=args.channels_last and device.type == "cuda",
            logger=logger,
        )
        if report is not None:
            fold_reports.append(report)

    if not fold_reports:
        raise RuntimeError("Cross-validation finished without any fold metrics.")

    macro_f1 = np.mean([report["macro avg"]["f1-score"] for report in fold_reports])
    accuracy = np.mean([report["accuracy"] for report in fold_reports])

    summary = (
        f"\nCV Summary ({args.model}, {args.folds}-fold, seed {args.seed})\n"
        f"Mean Accuracy: {accuracy:.4f}\n"
        f"Mean Macro F1: {macro_f1:.4f}\n"
    )

    logger.info(summary.strip())
    with CV_RESULTS.open("a", encoding="utf-8") as file:
        file.write(summary + "\n")

    append_fold_metrics(args, fold_reports)
    logger.info("Fold metrics appended to: %s", CV_METRICS_CSV)


if __name__ == "__main__":
    main()
