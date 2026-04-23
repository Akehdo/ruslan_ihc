import argparse
import csv
import sys
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
    get_device,
    get_model,
    get_transforms,
    set_seed,
    validate_dataset_dirs,
)


CV_RESULTS = OUTPUTS_DIR / "cv_results.txt"
CV_METRICS_CSV = OUTPUTS_DIR / "cv_metrics.csv"


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
    return parser.parse_args()


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, epochs) -> float:
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


def evaluate_epoch(model, val_loader, criterion, device) -> tuple[float, dict]:
    model.eval()
    running_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    val_loss = running_loss / len(val_loader)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return val_loss, report


def train_one_fold(model, train_loader, val_loader, device, epochs, lr, patience, best_path, last_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    patience_counter = 0
    best_report = None

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, epochs)
        val_loss, report = evaluate_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_report = report
            torch.save(model.state_dict(), best_path)
            print(f"New best model saved: {best_path}")
        else:
            patience_counter += 1
            print(f"No val improvement for {patience_counter} epoch(s)")

        torch.save(model.state_dict(), last_path)

        if patience_counter >= patience:
            print("Early stopping triggered")
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
    set_seed(args.seed)
    ensure_project_dirs()
    validate_dataset_dirs()

    device = get_device()
    pin_memory = device.type == "cuda"
    use_augmentation = not args.no_augmentation

    train_transform, val_transform = get_transforms(use_augmentation=use_augmentation)

    full_dataset = ImageFolder(TRAIN_DIR, transform=train_transform)
    targets = full_dataset.targets

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_reports = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(targets)), targets), 1):
        print(f"\n=== Fold {fold}/{args.folds} ===")

        train_subset = Subset(full_dataset, train_idx)
        val_dataset = ImageFolder(TRAIN_DIR, transform=val_transform)
        val_subset = Subset(val_dataset, val_idx)

        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )

        model = get_model(args.model).to(device)

        best_path = MODELS_DIR / f"{args.model}_cv_fold_{fold}_seed_{args.seed}_best.pth"
        last_path = MODELS_DIR / f"{args.model}_cv_fold_{fold}_seed_{args.seed}_last.pth"

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

    print(summary)
    with CV_RESULTS.open("a", encoding="utf-8") as file:
        file.write(summary + "\n")

    append_fold_metrics(args, fold_reports)
    print(f"Fold metrics appended to: {CV_METRICS_CSV}")


if __name__ == "__main__":
    main()
