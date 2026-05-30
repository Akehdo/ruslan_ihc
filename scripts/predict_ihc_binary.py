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


class ImageDataset(Dataset):
    def __init__(self, paths: list[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        return self.transform(image), str(path)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict IHC binary positive/negative labels")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Image file or folder")
    parser.add_argument("--out", default="outputs/ihc_binary_predictions.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(path)
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def load_model(checkpoint_path: Path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint["model_name"]
    model = get_model(model_name, num_classes=2, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def main():
    args = parse_args()
    image_paths = collect_images(Path(args.input))
    if not image_paths:
        raise RuntimeError(f"No images found in {args.input}")

    device = get_device()
    model, checkpoint = load_model(Path(args.checkpoint), device)
    _, transform = get_transforms(use_augmentation=False)
    loader = DataLoader(
        ImageDataset(image_paths, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    class_names = checkpoint.get("class_names", ["negative", "positive"])
    marker = checkpoint.get("marker", "unknown")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "marker", "prediction", "confidence", "prob_negative", "prob_positive"])
        with torch.no_grad():
            for images, paths in loader:
                probs = torch.softmax(model(images.to(device)), dim=1)
                positive_probs = probs[:, 1]
                predictions = (positive_probs >= args.threshold).long()

                for path, pred, prob_neg, prob_pos in zip(paths, predictions, probs[:, 0], probs[:, 1]):
                    pred_idx = int(pred.item())
                    confidence = prob_pos.item() if pred_idx == 1 else prob_neg.item()
                    writer.writerow(
                        [
                            path,
                            marker,
                            class_names[pred_idx],
                            f"{confidence:.6f}",
                            f"{prob_neg.item():.6f}",
                            f"{prob_pos.item():.6f}",
                        ]
                    )

    print(f"Predictions written to: {out_path}")


if __name__ == "__main__":
    main()
