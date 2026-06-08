from __future__ import annotations

import argparse
import csv
import itertools
import sys
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.append(str(ROOT))

from cloudcover.io import load_gt_mask, load_rgb
from cloudcover.metrics import metrics
from cloudcover.methods.hsv import predict_mask as hsv_mask
from cloudcover.methods.hyta import predict_mask as hyta_mask
from cloudcover.methods.hyta import predict_mask_imx385_adapted as hyta_imx385_adapted_mask
from cloudcover.methods.ndrb import predict_mask as ndrb_mask
from cloudcover.methods.rbr import predict_mask as rbr_mask
from cloudcover.methods.tiny_ucloudnet import predict_mask as tiny_ucloudnet_mask


DATASET_NAME = "swinyseg"
DEFAULT_SPLIT_CSV = REPO_ROOT / "splits" / "swinyseg" / "swinyseg_split_both.csv"
DEFAULT_MODEL_PATH = ROOT / "tiny-ucloudnet" / "models" / "tiny_ucloudnet_swinyseg.keras"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "parameter_search_swinyseg_val_originals.csv"
DEFAULT_BEST_CSV = REPO_ROOT / "parameter_search_swinyseg_val_originals_best.csv"


METHODS = {
    "hsv": hsv_mask,
    "rbr": rbr_mask,
    "ndrb": ndrb_mask,
    "hyta": hyta_mask,
    "imx385_adapted": hyta_imx385_adapted_mask,
    "tiny_ucloudnet": tiny_ucloudnet_mask,
}


PARAM_GRIDS = {
    "rbr": {
        "thr": [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
    },
    "ndrb": {
        "thr": [-0.35, -0.30, -0.25, -0.20, -0.15, -0.10, -0.05, 0.0],
    },
    "hsv": {
        "s_thr": [0.20, 0.25, 0.30, 0.35, 0.45, 0.50],
        "v_min": [0.15, 0.20, 0.25, 0.30, 0.35, 0.45],
        "blur": [0, 1, 3, 5, 7, 9],
    },
    "hyta": {
        "Tf": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
        "sigma_thr": [0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
    },
    "imx385_adapted": {
        "Tf": [0.05],
        "sigma_thr": [0.01],
        "threshold_radius_scale": [0.55],
        "prediction_radius_scale": [0.65],
        "bottom_boost": [0.125],
        "bottom_boost_start": [0.55],
        "bottom_boost_gate": [0.35],
        "low_okta_base_cf_max": [0.25],
        "low_okta_cap_okta": [1],
    },
    "tiny_ucloudnet": {
        "model_path": [str(DEFAULT_MODEL_PATH.resolve())],
        "img_size": [304],
        "thr": [0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90],
    },
} 


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run parameter search on the SWINySeg validation split using only "
            "original images. By default, both day and night validation images are used."
        )
    )
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--subset", choices=["both", "day", "night"], default="both")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        help="Methods to search: all, hsv, rbr, ndrb, hyta, imx385_adapted, tiny_ucloudnet.",
    )
    parser.add_argument("--rank-by", default="iou")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--best-csv", type=Path, default=DEFAULT_BEST_CSV)
    parser.add_argument(
        "--include-augmented",
        action="store_true",
        help="Use augmented images too. Default is validation originals only.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional quick-test limit before running a full search.",
    )
    return parser.parse_args()


def resolve_methods(methods: list[str]) -> list[str]:
    if methods == ["all"]:
        return ["hsv", "rbr", "ndrb", "hyta", "imx385_adapted", "tiny_ucloudnet"]

    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        raise ValueError(f"Unknown method(s): {unknown}. Available methods: {sorted(METHODS)}")

    return methods


def load_pairs_from_split_csv(
    csv_path: Path,
    split: str,
    subset: str,
    originals_only: bool,
    max_images: int | None,
) -> list[dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Fant ikke split-fil: {csv_path}")

    pairs = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] != split:
                continue
            if subset != "both" and row["time_of_day"] != subset:
                continue
            if originals_only and row["is_original"].lower() != "true":
                continue

            pairs.append(
                {
                    "name": row["image_name"],
                    "time_of_day": row["time_of_day"],
                    "original_id": row["original_id"],
                    "is_original": row["is_original"].lower() == "true",
                    "augmentation_type": row["augmentation_type"],
                    "img_path": Path(row["image_path"]),
                    "gt_path": Path(row["mask_path"]),
                }
            )

            if max_images is not None and len(pairs) >= max_images:
                break

    if not pairs:
        raise RuntimeError(
            f"Ingen bilder funnet for split={split}, subset={subset}, "
            f"originals_only={originals_only}"
        )
    return pairs


def print_dataset_stats(pairs: list[dict], split: str, subset: str, originals_only: bool) -> None:
    n_day = sum(1 for pair in pairs if pair["time_of_day"] == "day")
    n_night = sum(1 for pair in pairs if pair["time_of_day"] == "night")
    n_original = sum(1 for pair in pairs if pair["is_original"])
    print(f"Split: {split}")
    print(f"Subset: {subset}")
    print(f"Originals only: {originals_only}")
    print(f"Images: {len(pairs)} (original={n_original}, day={n_day}, night={n_night})")


def mean_of_dicts(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


def generate_param_combinations(param_grid: dict):
    keys = list(param_grid.keys())
    values = [param_grid[key] for key in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def evaluate_method_on_pairs(method_func, params: dict, pairs: list[dict]) -> tuple[dict, int]:
    rows = []
    for pair in pairs:
        rgb = load_rgb(pair["img_path"])
        gt = load_gt_mask(pair["gt_path"])
        pred = method_func(rgb, **params)
        rows.append(metrics(pred, gt))

    return mean_of_dicts(rows), len(rows)


def parameter_search_for_method(method_name: str, pairs: list[dict], rank_by: str) -> list[dict]:
    method_func = METHODS[method_name]
    param_grid = PARAM_GRIDS[method_name]

    results = []
    for params in generate_param_combinations(param_grid):
        avg_metrics, n_images = evaluate_method_on_pairs(method_func, params, pairs)
        if not avg_metrics:
            continue
        if rank_by not in avg_metrics:
            raise KeyError(f"Metric '{rank_by}' does not exist. Available: {sorted(avg_metrics)}")

        result = {
            "dataset": DATASET_NAME,
            "split": "val",
            "method": method_name,
            "n_images": n_images,
            "rank_metric": rank_by,
            "rank_value": avg_metrics[rank_by],
            "params": deepcopy(params),
        }
        result.update(avg_metrics)
        results.append(result)

    results.sort(key=lambda row: row["rank_value"], reverse=True)
    return results


def flatten_result(result: dict) -> dict:
    row = dict(result)
    params = row.pop("params", {})
    for key, value in params.items():
        row[key] = value
    return row


def save_results_to_csv(results: list[dict], output_path: Path) -> None:
    if not results:
        return

    rows = [flatten_result(result) for result in results]
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_top_results(method_name: str, results: list[dict], top_k: int) -> None:
    print(f"\nTop {min(top_k, len(results))} for {method_name}:")
    for idx, result in enumerate(results[:top_k], start=1):
        params = ", ".join(f"{key}={value}" for key, value in result["params"].items())
        print(
            f"{idx:2d}. {result['rank_metric']}={result['rank_value']:.4f} "
            f"accuracy={result.get('accuracy', float('nan')):.4f} "
            f"precision={result.get('precision', float('nan')):.4f} "
            f"recall={result.get('recall', float('nan')):.4f} "
            f"f1={result.get('f1', float('nan')):.4f} "
            f"dice={result.get('dice', float('nan')):.4f} "
            f"cloud_abs_error={result.get('cloud_abs_error', float('nan')):.4f} "
            f"params: {params}"
        )


def main() -> None:
    args = parse_args()
    method_names = resolve_methods(args.methods)
    originals_only = not args.include_augmented

    pairs = load_pairs_from_split_csv(
        args.split_csv,
        split=args.split,
        subset=args.subset,
        originals_only=originals_only,
        max_images=args.max_images,
    )

    print(f"Split CSV: {args.split_csv.resolve()}")
    print_dataset_stats(pairs, args.split, args.subset, originals_only)
    print(f"Methods: {', '.join(method_names)}")
    print(f"Ranking by: {args.rank_by}")

    all_results = []
    best_results = []

    for method_name in method_names:
        print("\n" + "=" * 70)
        print(f"Searching parameters for method: {method_name}")
        print("=" * 70)

        if method_name == "tiny_ucloudnet":
            model_path = PARAM_GRIDS["tiny_ucloudnet"]["model_path"][0]
            if not Path(model_path).exists():
                raise FileNotFoundError(
                    f"Tiny U Cloud Net model not found: {model_path}. "
                    "Train it first with SW/tiny-ucloudnet/train.py, or exclude "
                    "tiny_ucloudnet from --methods."
                )

        results = parameter_search_for_method(method_name, pairs, rank_by=args.rank_by)
        all_results.extend(results)
        if results:
            best_results.append(results[0])
        print_top_results(method_name, results, top_k=args.top_k)

    save_results_to_csv(all_results, args.output_csv)
    save_results_to_csv(best_results, args.best_csv)

    print("\n" + "=" * 70)
    print("Best parameter set per method")
    print("=" * 70)
    for result in best_results:
        print(f"{result['method']}: {result['rank_metric']}={result['rank_value']:.4f}, params={result['params']}")

    print(f"\nSaved all results:  {args.output_csv.resolve()}")
    print(f"Saved best results: {args.best_csv.resolve()}")


if __name__ == "__main__":
    main()
