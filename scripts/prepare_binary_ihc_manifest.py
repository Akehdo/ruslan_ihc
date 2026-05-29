import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ihc_binary import iter_markers, load_marker_records  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Create a manifest and summary for binary IHC datasets")
    parser.add_argument("--markers", nargs="+", default=["all"], help="Markers: Her2 ER PR Ki67 or all")
    parser.add_argument("--out", default="outputs/binary_ihc_manifest.csv")
    parser.add_argument("--summary", default="outputs/binary_ihc_summary.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    requested_markers = []
    for marker_arg in args.markers:
        requested_markers.extend(iter_markers(marker_arg))
    markers = list(dict.fromkeys(requested_markers))

    out_path = Path(args.out)
    summary_path = Path(args.summary)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    with out_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(["marker", "patient_id", "label", "label_name", "raw_label", "path"])

        for marker in markers:
            records = load_marker_records(marker)
            label_counts = Counter(record.label for record in records)
            patient_counts = Counter(record.patient_id for record in records)
            patient_label_counts = defaultdict(Counter)

            for record in records:
                label_name = "positive" if record.label == 1 else "negative"
                writer.writerow([marker, record.patient_id, record.label, label_name, record.raw_label, record.path])
                patient_label_counts[record.patient_id][record.label] += 1

            mixed_patients = sum(1 for counts in patient_label_counts.values() if len(counts) > 1)
            summary_rows.append({
                "marker": marker,
                "images": len(records),
                "patients": len(patient_counts),
                "negative_images": label_counts.get(0, 0),
                "positive_images": label_counts.get(1, 0),
                "mixed_label_patients": mixed_patients,
            })

    with summary_path.open("w", newline="", encoding="utf-8") as summary_file:
        fieldnames = [
            "marker",
            "images",
            "patients",
            "negative_images",
            "positive_images",
            "mixed_label_patients",
        ]
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Manifest written to: {out_path}")
    print(f"Summary written to: {summary_path}")
    for row in summary_rows:
        print(
            f"{row['marker']}: {row['images']} images, {row['patients']} patients, "
            f"negative={row['negative_images']}, positive={row['positive_images']}"
        )


if __name__ == "__main__":
    main()
