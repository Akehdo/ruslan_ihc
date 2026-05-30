import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image

try:
    from torch.utils.data import Dataset
except ImportError:  # Allows metadata utilities to run before the PyTorch env is repaired.
    Dataset = object


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "src" / "data"


MARKER_CONFIG = {
    "Her2": {
        "label_dir": "Labels_Her2",
        "image_dir": "Her2",
        "negative_label": "0",
        "positive_label": "1",
    },
    "ER": {
        "label_dir": "Labels_ER",
        "image_dir": "ER",
        "negative_label": "0",
        "positive_label": "a",
    },
    "PR": {
        "label_dir": "Labels_PR",
        "image_dir": "PR",
        "negative_label": "0",
        "positive_label": "a",
    },
    "Ki67": {
        "label_dir": "Labels_Ki67",
        "image_dir": "Ki67",
        "negative_label": "0",
        "positive_label": "a",
    },
}


@dataclass(frozen=True)
class PatchRecord:
    path: Path
    label: int
    patient_id: str
    marker: str
    raw_label: str


def normalize_marker(marker: str) -> str:
    aliases = {
        "her2": "Her2",
        "HER2": "Her2",
        "er": "ER",
        "pr": "PR",
        "pgr": "PR",
        "PGR": "PR",
        "ki67": "Ki67",
        "KI67": "Ki67",
    }
    return aliases.get(marker, marker)


def iter_markers(marker: str) -> Iterable[str]:
    if marker.lower() == "all":
        return MARKER_CONFIG.keys()
    normalized = normalize_marker(marker)
    if normalized not in MARKER_CONFIG:
        choices = ", ".join(["all", *MARKER_CONFIG.keys()])
        raise ValueError(f"Unknown marker '{marker}'. Expected one of: {choices}")
    return [normalized]


def load_marker_records(
    marker: str,
    data_dir: Path = DATA_DIR,
    require_exists: bool = True,
) -> list[PatchRecord]:
    marker = normalize_marker(marker)
    config = MARKER_CONFIG[marker]
    label_dir = data_dir / "Labels" / config["label_dir"]
    image_dir = data_dir / "Images" / "IHC" / config["image_dir"]
    label_map = {
        config["negative_label"]: 0,
        config["positive_label"]: 1,
    }

    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    records: list[PatchRecord] = []
    missing = 0
    skipped_labels = set()

    for csv_path in sorted(label_dir.glob("*.csv")):
        with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file, skipinitialspace=True)
            for row in reader:
                rel_image = (row.get("file_image") or "").strip()
                raw_label = (row.get("manual_annot") or "").strip()
                if not rel_image or raw_label not in label_map:
                    if raw_label:
                        skipped_labels.add(raw_label)
                    continue

                rel_path = Path(rel_image.replace("\\", "/")).with_suffix(".jpg")
                image_path = image_dir / rel_path
                if require_exists and not image_path.exists():
                    missing += 1
                    continue

                parts = rel_path.parts
                patient_id = parts[0] if parts else csv_path.stem
                records.append(
                    PatchRecord(
                        path=image_path,
                        label=label_map[raw_label],
                        patient_id=patient_id,
                        marker=marker,
                        raw_label=raw_label,
                    )
                )

    if not records:
        raise RuntimeError(f"No usable records found for marker {marker}. Missing files: {missing}")

    if missing:
        print(f"WARNING: skipped {missing} missing image paths for {marker}")
    if skipped_labels:
        print(f"WARNING: skipped unknown labels for {marker}: {sorted(skipped_labels)}")

    return records


def load_manifest_records(
    manifest_path: Path,
    marker: str,
    require_exists: bool = True,
) -> list[PatchRecord]:
    marker = normalize_marker(marker)
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    records: list[PatchRecord] = []
    missing = 0
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            row_marker = normalize_marker((row.get("marker") or "").strip())
            if row_marker != marker:
                continue

            path_text = (row.get("path") or "").strip().replace("\\", "/")
            if not path_text:
                continue

            image_path = Path(path_text)
            if not image_path.is_absolute():
                image_path = manifest_path.parent / image_path

            if require_exists and not image_path.exists():
                missing += 1
                continue

            records.append(
                PatchRecord(
                    path=image_path,
                    label=int(row["label"]),
                    patient_id=row["patient_id"],
                    marker=marker,
                    raw_label=row.get("raw_label", row["label"]),
                )
            )

    if not records:
        raise RuntimeError(f"No usable manifest records found for marker {marker}. Missing files: {missing}")
    if missing:
        print(f"WARNING: skipped {missing} missing manifest image paths for {marker}")
    return records


class IHCBinaryPatchDataset(Dataset):
    def __init__(self, records: list[PatchRecord], transform=None):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record.path) as opened:
            image = opened.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, record.label
