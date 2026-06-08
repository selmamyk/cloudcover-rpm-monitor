from __future__ import annotations

import argparse
import csv
import sys
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
DEFAULT_PARAMS_CSV = REPO_ROOT / "parameter_search_swinyseg_val_originals_best.csv"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "swinyseg"
DEFAULT_MODEL_PATH = ROOT / "tiny-ucloudnet" / "models" / "tiny_ucloudnet_swinyseg.keras"


METHODS = {
    "hsv": hsv_mask,
    "rbr": rbr_mask,
    "ndrb": ndrb_mask,
    "hyta": hyta_mask,
    "imx385_adapted": hyta_imx385_adapted_mask,
    "tiny_ucloudnet": tiny_ucloudnet_mask,
}


PARAM_COLUMNS = {
    "hsv": {"s_thr": float, "v_min": float, "blur": int},
    "rbr": {"thr": float},
    "ndrb": {"thr": float},
    "hyta": {"Tf": float, "sigma_thr": float},
    "imx385_adapted": {
        "Tf": float,
        "sigma_thr": float,
        "threshold_radius_scale": float,
        "prediction_radius_scale": float,
        "bottom_boost": float,
        "bottom_boost_start": float,
        "bottom_boost_gate": float,
        "low_okta_base_cf_max": float,
        "low_okta_cap_okta": int,
    },
    "tiny_ucloudnet": {"model_path": str, "img_size": int, "thr": float},
}


DEFAULT_PARAMS = {
    "hsv": {"s_thr": 0.35, "v_min": 0.30, "blur": 0},
    "rbr": {"thr": 0.60},
    "ndrb": {"thr": -0.25},
    "hyta": {"Tf": 0.20, "sigma_thr": 0.05},
    "imx385_adapted": {
        "Tf": 0.05,
        "sigma_thr": 0.01,
        "threshold_radius_scale": 0.55,
        "prediction_radius_scale": 0.65,
        "bottom_boost": 0.125,
        "bottom_boost_start": 0.55,
        "bottom_boost_gate": 0.35,
        "low_okta_base_cf_max": 0.25,
        "low_okta_cap_okta": 1,
    },
    "tiny_ucloudnet": {
        "model_path": str(DEFAULT_MODEL_PATH.resolve()),
        "img_size": 304,
        "thr": 0.60,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate SWINySeg methods on the held-out test split. The test set "
            "uses original images only by default."
        )
    )
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--params-csv", type=Path, default=DEFAULT_PARAMS_CSV)
    parser.add_argument("--subset", choices=["both", "day", "night"], default="both")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        help="Methods to evaluate: all, hsv, rbr, ndrb, hyta, imx385_adapted, tiny_ucloudnet.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional output path. Default includes the selected subset in the filename.",
    )
    parser.add_argument(
        "--per-image-csv",
        type=Path,
        default=None,
        help="Optional output path. Default includes the selected subset in the filename.",
    )
    parser.add_argument(
        "--use-default-params",
        action="store_true",
        help="Use built-in fallback parameters instead of reading --params-csv.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional quick-test limit before running full evaluation.",
    )
    return parser.parse_args()


def resolve_methods(methods: list[str]) -> list[str]:
    if methods == ["all"]:
        return ["hsv", "rbr", "ndrb", "hyta", "imx385_adapted", "tiny_ucloudnet"]

    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        raise ValueError(f"Unknown method(s): {unknown}. Available methods: {sorted(METHODS)}")

    return methods


def parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def load_pairs_from_split_csv(
    csv_path: Path,
    subset: str,
    max_images: int | None,
) -> list[dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Fant ikke split-fil: {csv_path}")

    pairs = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] != "test":
                continue
            if subset != "both" and row["time_of_day"] != subset:
                continue
            if not parse_bool(row["is_original"]):
                continue

            pairs.append(
                {
                    "name": row["image_name"],
                    "time_of_day": row["time_of_day"],
                    "original_id": row["original_id"],
                    "img_path": Path(row["image_path"]),
                    "gt_path": Path(row["mask_path"]),
                }
            )

            if max_images is not None and len(pairs) >= max_images:
                break

    if not pairs:
        raise RuntimeError(f"Ingen test-originalbilder funnet for subset={subset}")
    return pairs


def cast_param(value: str, caster):
    if caster is str:
        return value
    if caster is int:
        return int(float(value))
    return caster(value)


def load_best_params(params_csv: Path, method_names: list[str]) -> dict[str, dict]:
    if not params_csv.exists():
        raise FileNotFoundError(
            f"Fant ikke parameterfil: {params_csv}. "
            "Kjor parameter_search.py forst, eller bruk --use-default-params."
        )

    best_params = {}
    with params_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row.get("method", "")
            if method not in method_names:
                continue

            params = {}
            for column, caster in PARAM_COLUMNS[method].items():
                value = row.get(column, "")
                if value == "":
                    continue
                params[column] = cast_param(value, caster)

            merged = dict(DEFAULT_PARAMS[method])
            merged.update(params)
            best_params[method] = merged

    missing = [method for method in method_names if method not in best_params]
    if missing:
        raise RuntimeError(
            f"Parameterfilen mangler beste parametere for: {missing}. "
            "Kjor parameter_search.py med disse metodene, eller bruk --use-default-params."
        )

    return best_params


def mean_of_dicts(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


def evaluate_method(method_name: str, params: dict, pairs: list[dict]) -> tuple[dict, list[dict]]:
    method_func = METHODS[method_name]
    per_image_rows = []
    metric_rows = []

    for pair in pairs:
        rgb = load_rgb(pair["img_path"])
        gt = load_gt_mask(pair["gt_path"])
        pred = method_func(rgb, **params)
        row_metrics = metrics(pred, gt)
        metric_rows.append(row_metrics)

        per_image_rows.append(
            {
                "dataset": DATASET_NAME,
                "split": "test",
                "method": method_name,
                "image_name": pair["name"],
                "time_of_day": pair["time_of_day"],
                "original_id": pair["original_id"],
                **row_metrics,
            }
        )

    return mean_of_dicts(metric_rows), per_image_rows


def flatten_summary(method_name: str, subset: str, params: dict, n_images: int, avg_metrics: dict) -> dict:
    row = {
        "dataset": DATASET_NAME,
        "split": "test",
        "subset": subset,
        "originals_only": True,
        "method": method_name,
        "n_images": n_images,
    }
    row.update(params)
    row.update(avg_metrics)
    return row


def write_csv(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return

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


def print_dataset_stats(pairs: list[dict], subset: str) -> None:
    n_day = sum(1 for pair in pairs if pair["time_of_day"] == "day")
    n_night = sum(1 for pair in pairs if pair["time_of_day"] == "night")
    print(f"Subset: {subset}")
    print(f"Test original images: {len(pairs)} (day={n_day}, night={n_night})")


def print_summary(row: dict) -> None:
    print(
        f"{row['method']}: "
        f"iou={row.get('iou', float('nan')):.4f}, "
        f"dice={row.get('dice', float('nan')):.4f}, "
        f"f1={row.get('f1', float('nan')):.4f}, "
        f"accuracy={row.get('accuracy', float('nan')):.4f}, "
        f"cloud_abs_error={row.get('cloud_abs_error', float('nan')):.4f}"
    )


def default_output_paths(subset: str) -> tuple[Path, Path]:
    subset_name = "combined" if subset == "both" else subset
    summary_csv = DEFAULT_RESULTS_DIR / f"test_results_swinyseg_{subset_name}_originals_summary.csv"
    per_image_csv = DEFAULT_RESULTS_DIR / f"test_results_swinyseg_{subset_name}_originals_per_image.csv"
    return summary_csv, per_image_csv


def main() -> None:
    args = parse_args()
    method_names = resolve_methods(args.methods)
    pairs = load_pairs_from_split_csv(args.split_csv, subset=args.subset, max_images=args.max_images)
    default_summary_csv, default_per_image_csv = default_output_paths(args.subset)
    summary_csv = args.summary_csv or default_summary_csv
    per_image_csv = args.per_image_csv or default_per_image_csv

    if args.use_default_params:
        params_by_method = {method: dict(DEFAULT_PARAMS[method]) for method in method_names}
    else:
        params_by_method = load_best_params(args.params_csv, method_names)

    print(f"Split CSV: {args.split_csv.resolve()}")
    print_dataset_stats(pairs, args.subset)
    print(f"Methods: {', '.join(method_names)}")
    if args.use_default_params:
        print("Parameters: built-in defaults")
    else:
        print(f"Parameters: {args.params_csv.resolve()}")

    summary_rows = []
    per_image_rows = []

    for method_name in method_names:
        params = params_by_method[method_name]
        if method_name == "tiny_ucloudnet" and not Path(params["model_path"]).exists():
            raise FileNotFoundError(f"Tiny U Cloud Net model not found: {params['model_path']}")

        print("\n" + "=" * 70)
        print(f"Evaluating method: {method_name}")
        print(f"Params: {params}")
        print("=" * 70)

        avg_metrics, method_per_image_rows = evaluate_method(method_name, params, pairs)
        summary_row = flatten_summary(method_name, args.subset, params, len(pairs), avg_metrics)
        summary_rows.append(summary_row)
        per_image_rows.extend(method_per_image_rows)
        print_summary(summary_row)

    write_csv(summary_rows, summary_csv)
    write_csv(per_image_rows, per_image_csv)

    print("\nSaved summary results:")
    print(summary_csv.resolve())
    print("Saved per-image results:")
    print(per_image_csv.resolve())


if __name__ == "__main__":
    main()
