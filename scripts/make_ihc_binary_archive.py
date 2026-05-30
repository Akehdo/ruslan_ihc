import argparse
import csv
import shutil
import sys
import tarfile
from collections import Counter
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ihc_binary import iter_markers, load_marker_records  # noqa: E402


LABEL_NAMES = {
    0: "negative",
    1: "positive",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare labeled IHC images as a clean binary dataset and create a tar archive."
    )
    parser.add_argument("--markers", nargs="+", default=["all"], help="Markers: Her2 ER PR Ki67 or all")
    parser.add_argument("--out", default="outputs/ihc_binary_dataset", help="Prepared dataset directory")
    parser.add_argument("--archive", default=None, help="Archive path. Default: <out>.tar")
    parser.add_argument("--overwrite", action="store_true", help="Delete the output directory before rebuilding")
    return parser.parse_args()


def assert_readable_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.load()
        return True
    except Exception:
        return False


def prepared_filename(record) -> str:
    # Preserve patient/subregion context while flattening into class folders.
    return "__".join(record.path.parts[-3:]).replace("\\", "__").replace("/", "__")


def make_archive(dataset_dir: Path, archive_path: Path):
    if archive_path.exists():
        archive_path.unlink()

    print(f"Creating archive: {archive_path}")
    with tarfile.open(archive_path, "w") as tar:
        tar.add(dataset_dir, arcname=dataset_dir.name, recursive=True)


def main():
    args = parse_args()
    dataset_dir = Path(args.out)
    archive_path = Path(args.archive) if args.archive else dataset_dir.with_suffix(".tar")
    images_dir = dataset_dir / "images"
    manifest_path = dataset_dir / "manifest.csv"
    summary_path = dataset_dir / "summary.csv"
    bad_images_path = dataset_dir / "bad_images.csv"

    if dataset_dir.exists() and args.overwrite:
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    markers = []
    for marker in args.markers:
        markers.extend(iter_markers(marker))
    markers = list(dict.fromkeys(markers))

    summary_rows = []
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file, bad_images_path.open(
        "w", newline="", encoding="utf-8"
    ) as bad_file:
        manifest_writer = csv.writer(manifest_file)
        bad_writer = csv.writer(bad_file)
        manifest_writer.writerow(["marker", "patient_id", "label", "label_name", "raw_label", "path"])
        bad_writer.writerow(["marker", "patient_id", "label", "raw_label", "source_path"])

        for marker in markers:
            records = load_marker_records(marker)
            counts = Counter()
            patients = set()
            bad_count = 0
            print(f"Preparing {marker}: {len(records)} images")

            for index, record in enumerate(records, 1):
                if not assert_readable_image(record.path):
                    bad_count += 1
                    bad_writer.writerow([record.marker, record.patient_id, record.label, record.raw_label, record.path])
                    continue

                label_name = LABEL_NAMES[record.label]
                target_path = images_dir / marker / label_name / prepared_filename(record)
                target_path.parent.mkdir(parents=True, exist_ok=True)
                if not target_path.exists():
                    shutil.copy2(record.path, target_path)

                manifest_writer.writerow(
                    [
                        marker,
                        record.patient_id,
                        record.label,
                        label_name,
                        record.raw_label,
                        target_path.relative_to(dataset_dir).as_posix(),
                    ]
                )
                counts[record.label] += 1
                patients.add(record.patient_id)

                if index % 1000 == 0:
                    print(f"  {marker}: {index}/{len(records)} processed")

            summary_rows.append(
                {
                    "marker": marker,
                    "images": sum(counts.values()),
                    "patients": len(patients),
                    "negative_images": counts.get(0, 0),
                    "positive_images": counts.get(1, 0),
                    "bad_images": bad_count,
                }
            )
            print(
                f"Done {marker}: {sum(counts.values())} usable, "
                f"negative={counts.get(0, 0)}, positive={counts.get(1, 0)}, bad={bad_count}"
            )

    with summary_path.open("w", newline="", encoding="utf-8") as summary_file:
        fieldnames = ["marker", "images", "patients", "negative_images", "positive_images", "bad_images"]
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    make_archive(dataset_dir, archive_path)
    print(f"Prepared dataset: {dataset_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")
    print(f"Bad images: {bad_images_path}")
    print(f"Archive: {archive_path}")


if __name__ == "__main__":
    main()
