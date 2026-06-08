from __future__ import annotations

import argparse
import csv
import random
import re
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = REPO_ROOT / "datasets" / "swinyseg"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "splits" / "swinyseg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a leakage-safe SWINySeg split CSV. The split is made at "
            "original-image level, and augmented variants follow their original."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument(
        "--subset",
        choices=["both", "day", "night"],
        default="both",
        help="Limit the split to day, night, or keep both.",
    )
    return parser.parse_args()


def image_time_of_day(filename: str) -> str:
    stem = Path(filename).stem.lower()
    if stem.startswith("d"):
        return "day"
    if stem.startswith("n"):
        return "night"
    raise ValueError(f"Could not infer day/night from filename: {filename}")


def original_id(filename: str) -> str:
    stem = Path(filename).stem.lower()
    match = re.match(r"^([dn]\d+)(?:_\d+)?$", stem)
    if not match:
        raise ValueError(f"Could not infer original_id from filename: {filename}")
    return match.group(1)


def augmentation_type(filename: str) -> str:
    stem = Path(filename).stem.lower()
    match = re.match(r"^[dn]\d+_(\d+)$", stem)
    if match:
        return f"aug_{match.group(1)}"
    return "none"


def is_original(filename: str) -> bool:
    return augmentation_type(filename) == "none"


def read_source_metadata(dataset_root: Path) -> dict[str, dict[str, str]]:
    metadata_path = dataset_root / "metadata.csv"
    if not metadata_path.exists():
        return {}

    with metadata_path.open("r", newline="", encoding="utf-8") as f:
        return {row["Name"]: row for row in csv.DictReader(f)}


def collect_rows(dataset_root: Path, subset: str) -> list[dict[str, str]]:
    images_dir = dataset_root / "images"
    masks_dir = dataset_root / "GTmaps"
    source_metadata = read_source_metadata(dataset_root)

    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Missing mask directory: {masks_dir}")

    rows = []
    for image_path in sorted(images_dir.glob("*.jpg")):
        mask_path = masks_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path}")

        time_of_day = image_time_of_day(image_path.name)
        if subset != "both" and time_of_day != subset:
            continue

        oid = original_id(image_path.name)
        original_name = f"{oid}.jpg"
        metadata = source_metadata.get(original_name, {})

        rows.append(
            {
                "image_name": image_path.name,
                "mask_name": mask_path.name,
                "image_path": str(image_path.resolve()),
                "mask_path": str(mask_path.resolve()),
                "original_id": oid,
                "is_original": str(is_original(image_path.name)).lower(),
                "augmentation_type": augmentation_type(image_path.name),
                "time_of_day": time_of_day,
                "date": metadata.get("Date", ""),
                "time": metadata.get("Time", ""),
                "fnumber": metadata.get("Fnumber", ""),
                "exposure_time": metadata.get("ExposureTime", ""),
                "iso": metadata.get("ISO", ""),
            }
        )

    if not rows:
        raise RuntimeError(f"No image/mask pairs found in {dataset_root} for subset={subset}")
    return rows


def split_original_ids(
    rows: list[dict[str, str]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, str]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError("train, val and test ratios must sum to 1.0")

    originals_by_time: dict[str, list[str]] = {"day": [], "night": []}
    seen = set()
    for row in rows:
        if row["is_original"] != "true":
            continue
        oid = row["original_id"]
        if oid in seen:
            raise ValueError(f"Duplicate original image found for original_id={oid}")
        seen.add(oid)
        originals_by_time[row["time_of_day"]].append(oid)

    rng = random.Random(seed)
    split_by_original_id = {}

    for time_of_day, ids in originals_by_time.items():
        if not ids:
            continue

        ids = sorted(ids)
        rng.shuffle(ids)

        n_total = len(ids)
        n_train = round(n_total * train_ratio)
        n_val = round(n_total * val_ratio)

        for oid in ids[:n_train]:
            split_by_original_id[oid] = "train"
        for oid in ids[n_train : n_train + n_val]:
            split_by_original_id[oid] = "val"
        for oid in ids[n_train + n_val :]:
            split_by_original_id[oid] = "test"

    missing_originals = sorted({row["original_id"] for row in rows} - set(split_by_original_id))
    if missing_originals:
        raise RuntimeError(
            "Some augmented images do not have a matching original image: "
            + ", ".join(missing_originals[:10])
        )

    return split_by_original_id


def add_split(rows: list[dict[str, str]], split_by_original_id: dict[str, str]) -> list[dict[str, str]]:
    split_rows = []
    for row in rows:
        row = dict(row)
        row["split"] = split_by_original_id[row["original_id"]]
        split_rows.append(row)
    return sorted(split_rows, key=lambda r: (r["split"], r["time_of_day"], r["original_id"], r["image_name"]))


def validate_no_leakage(rows: list[dict[str, str]]) -> None:
    splits_by_original_id: dict[str, set[str]] = {}
    for row in rows:
        splits_by_original_id.setdefault(row["original_id"], set()).add(row["split"])

    leaked = {oid: splits for oid, splits in splits_by_original_id.items() if len(splits) > 1}
    if leaked:
        example = next(iter(leaked.items()))
        raise RuntimeError(f"Data leakage detected for {example[0]} across splits {sorted(example[1])}")


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_name",
        "mask_name",
        "image_path",
        "mask_path",
        "original_id",
        "is_original",
        "augmentation_type",
        "time_of_day",
        "split",
        "date",
        "time",
        "fnumber",
        "exposure_time",
        "iso",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_stats(rows: list[dict[str, str]]) -> None:
    print("Split statistics:")
    for split in ["train", "val", "test"]:
        split_rows = [row for row in rows if row["split"] == split]
        original_rows = [row for row in split_rows if row["is_original"] == "true"]
        by_time = Counter(row["time_of_day"] for row in split_rows)
        original_by_time = Counter(row["time_of_day"] for row in original_rows)
        print(
            f"  {split:5s}: "
            f"{len(split_rows):4d} images, {len(original_rows):4d} originals "
            f"(day={original_by_time['day']}, night={original_by_time['night']} originals; "
            f"all rows day={by_time['day']}, night={by_time['night']})"
        )


def main() -> None:
    args = parse_args()
    rows = collect_rows(args.dataset_root, args.subset)
    split_by_original_id = split_original_ids(
        rows,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    rows = add_split(rows, split_by_original_id)
    validate_no_leakage(rows)

    split_csv = args.output_dir / f"swinyseg_split_{args.subset}.csv"
    write_csv(rows, split_csv)

    print(f"Dataset root: {args.dataset_root.resolve()}")
    print(f"Output file:  {split_csv.resolve()}")
    print(f"Seed:         {args.seed}")
    print_stats(rows)


if __name__ == "__main__":
    main()
