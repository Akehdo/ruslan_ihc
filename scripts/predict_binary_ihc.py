import argparse
import csv
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common import get_device, get_model, get_transforms  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


class ImageOnlyDataset(Dataset):
    def __init__(self, image_paths: list[Path], transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, str(path)


def parse_args():
    parser = argparse.ArgumentParser(description="Run binary IHC checkpoint inference on images")
    parser.add_argument("--checkpoint", required=True, help="Path to binary_*_best.pth checkpoint")
    parser.add_argument("--input", required=True, help="Image file or directory with images")
    parser.add_argument("--out", default="outputs/binary_ihc_predictions.csv", help="Output CSV path")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5, help="Positive-class probability threshold")
    parser.add_argument("--model", default=None, help="Fallback model name if checkpoint has no metadata")
    return parser.parse_args()


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input not found: {path}")
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def load_checkpoint(path: Path, fallback_model: str | None, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_name = checkpoint.get("model_name") or fallback_model
        marker = checkpoint.get("marker", "unknown")
        class_names = checkpoint.get("class_names", ["negative", "positive"])
        state_dict = checkpoint["model_state_dict"]
    else:
        if fallback_model is None:
            raise ValueError("Checkpoint has no metadata. Pass --model, e.g. --model convnext")
        model_name = fallback_model
        marker = "unknown"
        class_names = ["negative", "positive"]
        state_dict = checkpoint

    model = get_model(model_name, num_classes=2, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, model_name, marker, class_names


def main():
    args = parse_args()
    input_path = Path(args.input)
    out_path = Path(args.out)
    checkpoint_path = Path(args.checkpoint)
    image_paths = collect_images(input_path)
    if not image_paths:
        raise RuntimeError(f"No images found in: {input_path}")

    device = get_device()
    model, model_name, marker, class_names = load_checkpoint(checkpoint_path, args.model, device)
    _, eval_transform = get_transforms(use_augmentation=False)
    dataset = ImageOnlyDataset(image_paths, transform=eval_transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "image",
            "marker",
            "model",
            "prediction",
            "confidence",
            "prob_negative",
            "prob_positive",
        ])

        with torch.no_grad():
            for images, paths in loader:
                images = images.to(device, non_blocking=True)
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
                positive_probs = probs[:, 1]
                predictions = (positive_probs >= args.threshold).long()

                for path, pred, prob_neg, prob_pos in zip(paths, predictions, probs[:, 0], probs[:, 1]):
                    pred_idx = int(pred.item())
                    confidence = prob_pos.item() if pred_idx == 1 else prob_neg.item()
                    writer.writerow([
                        path,
                        marker,
                        model_name,
                        class_names[pred_idx],
                        f"{confidence:.6f}",
                        f"{prob_neg.item():.6f}",
                        f"{prob_pos.item():.6f}",
                    ])

    print(f"Predictions written to: {out_path}")


if __name__ == "__main__":
    main()
