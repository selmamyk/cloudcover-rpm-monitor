#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "sorted_IMX385_dataset"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "hyta_imx385_variants"
EXPORT_CAPTURE_HYTA = REPO_ROOT / "IMX385_dataset" / "export_capture_hyta.py"
OKTA_RE = re.compile(r"^([0-8])_oktas$")
MODES = ("hyta", "low_okta_guard")


def load_export_capture_hyta():
    spec = importlib.util.spec_from_file_location("export_capture_hyta", EXPORT_CAPTURE_HYTA)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {EXPORT_CAPTURE_HYTA}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_modes(value: str) -> list[str]:
    modes = [part.strip() for part in value.split(",") if part.strip()]
    invalid = [mode for mode in modes if mode not in MODES]
    if invalid:
        allowed = ", ".join(MODES)
        raise argparse.ArgumentTypeError(f"Unknown mode(s): {', '.join(invalid)}. Allowed modes: {allowed}")
    return modes


def parse_oktas(value: str) -> set[int]:
    oktas = {int(part.strip()) for part in value.split(",") if part.strip()}
    invalid = sorted(okta for okta in oktas if okta < 0 or okta > 8)
    if invalid:
        raise argparse.ArgumentTypeError(f"Okta values must be 0..8, got: {invalid}")
    return oktas


def iter_samples(dataset_root: Path, split: str):
    split_root = dataset_root / split
    if not split_root.exists():
        raise FileNotFoundError(f"Split folder not found: {split_root}")
    for okta_dir in sorted(split_root.iterdir(), key=lambda path: path.name):
        match = OKTA_RE.match(okta_dir.name)
        if not okta_dir.is_dir() or not match:
            continue
        true_okta = int(match.group(1))
        for sample_dir in sorted(okta_dir.iterdir(), key=lambda path: path.name):
            raw_path = sample_dir / "image.raw"
            metadata_path = sample_dir / "metadata.json"
            if sample_dir.is_dir() and raw_path.exists() and metadata_path.exists():
                yield {
                    "true_okta": true_okta,
                    "sample_dir": sample_dir,
                    "raw_path": raw_path,
                    "metadata_path": metadata_path,
                    "sample_key": f"{split}_{true_okta}_oktas_{sample_dir.name}",
                }


def read_size(metadata_path: Path, width: int, height: int) -> tuple[int, int]:
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return int(metadata.get("width", width)), int(metadata.get("height", height))


def load_preview_bgr(hyta, sample: dict, args: argparse.Namespace) -> np.ndarray:
    if args.source == "preview":
        image_path = sample["sample_dir"] / args.image_name
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read preview image: {image_path}")
        return bgr

    width, height = read_size(sample["metadata_path"], args.width, args.height)
    raw10 = hyta.unpack_raw10(sample["raw_path"], width, height)
    return hyta.raw10_to_preview_bgr(raw10, bayer_pattern=args.bayer_pattern)


def valid_masks(hyta, shape: tuple[int, int], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    prediction = hyta.build_center_mask(height, width, args.prediction_radius_scale).astype(bool)
    threshold = hyta.build_center_mask(height, width, args.threshold_radius_scale).astype(bool)
    return prediction, threshold


def threshold_for_values(hyta, values: np.ndarray, tf: float, sigma_thr: float, levels: int) -> tuple[float, float]:
    if values.size == 0:
        return float(tf), 0.0
    sigma = float(values.std())
    if sigma < sigma_thr:
        return float(tf), sigma
    t_star = hyta.mce_threshold(hyta.to_levels(values, levels=levels), levels=levels)
    threshold = (t_star / (levels - 1)) * 2.0 - 1.0
    return float(threshold), sigma


def predict_hyta(hyta, rgb: np.ndarray, prediction_mask: np.ndarray, threshold_mask: np.ndarray, args: argparse.Namespace):
    lamn = hyta.normalized_br_ratio(rgb)
    threshold, sigma = threshold_for_values(
        hyta,
        lamn[threshold_mask],
        tf=args.hyta_tf,
        sigma_thr=args.hyta_sigma_thr,
        levels=args.levels,
    )
    mask = (lamn < threshold) & prediction_mask
    return mask, threshold, sigma, lamn


def predict_bottom_boost(lamn: np.ndarray, prediction_mask: np.ndarray, threshold: float, args: argparse.Namespace):
    height, _width = lamn.shape
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    ramp = np.clip(
        (y - float(args.bottom_boost_start)) / max(1.0 - float(args.bottom_boost_start), 1e-6),
        0.0,
        1.0,
    )
    threshold_map = np.broadcast_to(threshold + args.bottom_boost * ramp, lamn.shape)
    return (lamn < threshold_map) & prediction_mask, float(np.mean(threshold_map[prediction_mask]))


def cap_mask_fraction(mask: np.ndarray, lamn: np.ndarray, prediction_mask: np.ndarray, max_fraction: float) -> np.ndarray:
    max_pixels = int(round(float(max_fraction) * int(prediction_mask.sum())))
    if max_pixels <= 0:
        return np.zeros_like(mask, dtype=bool)
    cloud_positions = np.flatnonzero(mask & prediction_mask)
    if cloud_positions.size <= max_pixels:
        return mask & prediction_mask

    flat_lamn = lamn.reshape(-1)
    keep_order = np.argpartition(flat_lamn[cloud_positions], max_pixels - 1)[:max_pixels]
    capped = np.zeros_like(mask, dtype=bool).reshape(-1)
    capped[cloud_positions[keep_order]] = True
    return capped.reshape(mask.shape)


def predict_low_okta_guard(
    base_mask: np.ndarray,
    base_thr: float,
    base_sigma: float,
    lamn: np.ndarray,
    prediction_mask: np.ndarray,
    args: argparse.Namespace,
):
    base_cloud_fraction = float(base_mask[prediction_mask].mean()) if np.any(prediction_mask) else 0.0
    if base_cloud_fraction >= args.bottom_boost_gate:
        mask, threshold = predict_bottom_boost(lamn, prediction_mask, base_thr, args)
    else:
        mask, threshold = base_mask, base_thr

    if base_cloud_fraction <= args.low_okta_base_cf_max:
        mask = cap_mask_fraction(mask, lamn, prediction_mask, args.low_okta_cap_okta / 8.0)

    return mask, threshold, base_sigma


def draw_output(
    bgr: np.ndarray,
    mask: np.ndarray,
    prediction_mask: np.ndarray,
    label: str,
    out_prefix: Path,
) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    overlay = bgr.astype(np.float32).copy()
    red = np.zeros_like(overlay)
    red[..., 2] = 255.0
    overlay[mask] = 0.62 * overlay[mask] + 0.38 * red[mask]
    contours, _ = cv2.findContours(prediction_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    mask_panel = np.zeros_like(bgr)
    mask_panel[mask] = (255, 255, 255)
    debug = np.hstack([bgr, overlay, mask_panel])
    for image in (overlay, debug):
        cv2.rectangle(image, (0, 0), (image.shape[1], 42), (0, 0, 0), thickness=-1)
        cv2.putText(image, label, (18, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(str(out_prefix.with_name(out_prefix.name + "_overlay.png")), overlay)
    cv2.imwrite(str(out_prefix.with_name(out_prefix.name + "_mask.png")), (mask.astype(np.uint8) * 255))
    cv2.imwrite(str(out_prefix.with_name(out_prefix.name + "_debug.png")), debug)
    save_thumbnail(out_prefix.with_name(out_prefix.name + "_debug.png"), out_prefix.with_name(out_prefix.name + "_thumb.jpg"))


def save_thumbnail(src_path: Path, dst_path: Path, width: int = 900) -> None:
    image = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if image is None:
        return
    scale = width / max(float(image.shape[1]), 1.0)
    height = max(1, int(round(image.shape[0] * scale)))
    thumb = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst_path), thumb)


def rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_by_mode(rows: list[dict]) -> list[dict]:
    summaries = []
    for mode in sorted({row["mode"] for row in rows}):
        group = [row for row in rows if row["mode"] == mode]
        errors = [int(row["abs_okta_error"]) for row in group]
        summaries.append(
            {
                "mode": mode,
                "n_images": len(group),
                "accuracy": sum(error == 0 for error in errors) / len(errors),
                "within_1_okta": sum(error <= 1 for error in errors) / len(errors),
                "mean_abs_okta_error": sum(errors) / len(errors),
                "max_abs_okta_error": max(errors),
                "mean_cloud_fraction": float(np.mean([float(row["cloud_fraction"]) for row in group])),
            }
        )
    return summaries


def summarize_by_mode_and_okta(rows: list[dict]) -> list[dict]:
    summaries = []
    for mode in sorted({row["mode"] for row in rows}):
        mode_rows = [row for row in rows if row["mode"] == mode]
        for okta in range(9):
            group = [row for row in mode_rows if int(row["true_okta"]) == okta]
            if not group:
                continue
            errors = [int(row["abs_okta_error"]) for row in group]
            summaries.append(
                {
                    "mode": mode,
                    "true_okta": okta,
                    "n_images": len(group),
                    "accuracy": sum(error == 0 for error in errors) / len(errors),
                    "within_1_okta": sum(error <= 1 for error in errors) / len(errors),
                    "mean_abs_okta_error": sum(errors) / len(errors),
                    "mean_pred_okta": float(np.mean([int(row["pred_okta"]) for row in group])),
                    "mean_cloud_fraction": float(np.mean([float(row["cloud_fraction"]) for row in group])),
                }
            )
    return summaries


def confusion_rows(rows: list[dict]) -> list[dict]:
    output = []
    for mode in sorted({row["mode"] for row in rows}):
        mode_rows = [row for row in rows if row["mode"] == mode]
        for true_okta in range(9):
            row = {"mode": mode, "true_okta": true_okta}
            for pred_okta in range(9):
                row[f"pred_{pred_okta}"] = sum(
                    1
                    for item in mode_rows
                    if int(item["true_okta"]) == true_okta and int(item["pred_okta"]) == pred_okta
                )
            output.append(row)
    return output


def print_summary_table(summaries: list[dict]) -> None:
    if not summaries:
        return
    ranked = sorted(
        summaries,
        key=lambda row: (
            float(row["mean_abs_okta_error"]),
            -float(row["within_1_okta"]),
            -float(row["accuracy"]),
        ),
    )
    print("\n=== Variant summary, ranked by mean absolute okta error ===")
    print("mode              n    acc   within1   MAE   maxerr   mean_cloud")
    for row in ranked:
        print(
            f"{row['mode']:<16} "
            f"{int(row['n_images']):>4}  "
            f"{float(row['accuracy']):>5.3f}  "
            f"{float(row['within_1_okta']):>7.3f}  "
            f"{float(row['mean_abs_okta_error']):>5.3f}  "
            f"{int(row['max_abs_okta_error']):>6}  "
            f"{float(row['mean_cloud_fraction']):>10.3f}"
        )


def write_index(path: Path, rows: list[dict]) -> None:
    data_json = json.dumps(rows, ensure_ascii=False)
    mode_options = "".join(f'<option value="{mode}">{mode}</option>' for mode in sorted({row["mode"] for row in rows}))
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HYTA IMX385 test</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Arial, sans-serif; background: #eeeeea; color: #151515; }}
    header {{ display: grid; gap: 10px; padding: 12px 14px; background: #fff; border-bottom: 1px solid #d0d0ca; }}
    h1 {{ margin: 0; font-size: 18px; }}
    .toolbar {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
    button, select {{ height: 34px; border: 1px solid #b8b8b0; background: #fff; border-radius: 6px; padding: 0 10px; font: inherit; }}
    main {{ display: grid; grid-template-columns: minmax(0, 1fr) 330px; gap: 12px; padding: 12px; }}
    .stage {{ min-height: calc(100vh - 118px); display: grid; place-items: center; background: #111; border-radius: 6px; overflow: hidden; }}
    .stage img {{ display: block; max-width: 100%; max-height: calc(100vh - 136px); }}
    aside {{ display: grid; align-content: start; gap: 10px; }}
    .panel {{ background: #fff; border: 1px solid #d0d0ca; border-radius: 6px; padding: 12px; }}
    .kv {{ display: grid; grid-template-columns: 120px 1fr; gap: 6px 10px; font-size: 14px; }}
    .label {{ color: #666; }}
    .links {{ display: grid; gap: 8px; }}
    .strip {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }}
    .strip button {{ height: auto; padding: 0; overflow: hidden; }}
    .strip img {{ display: block; width: 100%; }}
    a {{ color: #064f87; }}
    @media (max-width: 900px) {{ main {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>HYTA IMX385 test</h1>
    <div class="toolbar">
      <button id="prev">Previous</button>
      <button id="next">Next</button>
      <span id="counter"></span>
      <select id="mode">{mode_options}</select>
      <select id="okta">
        <option value="all">All true oktas</option>
        <option value="0">0 oktas</option><option value="1">1 okta</option><option value="2">2 oktas</option>
        <option value="3">3 oktas</option><option value="4">4 oktas</option><option value="5">5 oktas</option>
        <option value="6">6 oktas</option><option value="7">7 oktas</option><option value="8">8 oktas</option>
      </select>
      <select id="sort">
        <option value="dataset">Dataset order</option>
        <option value="error">Largest error first</option>
        <option value="cloud_desc">Most cloud first</option>
        <option value="cloud_asc">Least cloud first</option>
      </select>
    </div>
  </header>
  <main>
    <section class="stage"><img id="image" alt=""></section>
    <aside>
      <section class="panel"><div class="kv">
        <span class="label">Mode</span><strong id="modeText"></strong>
        <span class="label">Sample</span><strong id="sample"></strong>
        <span class="label">True okta</span><span id="trueOkta"></span>
        <span class="label">Pred okta</span><span id="predOkta"></span>
        <span class="label">Error</span><span id="error"></span>
        <span class="label">Cloud</span><span id="cloud"></span>
        <span class="label">Threshold</span><span id="threshold"></span>
        <span class="label">Sigma</span><span id="sigma"></span>
      </div></section>
      <section class="panel links">
        <a id="debugLink" href="">Open debug image</a>
        <a id="overlayLink" href="">Open overlay image</a>
        <a id="maskLink" href="">Open mask image</a>
      </section>
      <section class="panel strip" id="strip"></section>
    </aside>
  </main>
  <script>
    const rows = {data_json};
    let filtered = [];
    let index = 0;
    const image = document.getElementById('image');
    const mode = document.getElementById('mode');
    const okta = document.getElementById('okta');
    const sort = document.getElementById('sort');
    const strip = document.getElementById('strip');

    function applyFilter() {{
      filtered = rows.filter(row => row.mode === mode.value && (okta.value === 'all' || String(row.true_okta) === okta.value));
      if (sort.value === 'error') filtered.sort((a, b) => b.abs_okta_error - a.abs_okta_error);
      if (sort.value === 'cloud_desc') filtered.sort((a, b) => b.cloud_fraction - a.cloud_fraction);
      if (sort.value === 'cloud_asc') filtered.sort((a, b) => a.cloud_fraction - b.cloud_fraction);
      index = Math.min(index, Math.max(filtered.length - 1, 0));
      render();
    }}
    function setIndex(nextIndex) {{
      if (!filtered.length) return;
      index = (nextIndex + filtered.length) % filtered.length;
      render();
    }}
    function renderStrip() {{
      strip.innerHTML = '';
      for (const offset of [-1, 0, 1]) {{
        if (!filtered.length) break;
        const itemIndex = (index + offset + filtered.length) % filtered.length;
        const item = filtered[itemIndex];
        const button = document.createElement('button');
        button.onclick = () => setIndex(itemIndex);
        const img = document.createElement('img');
        img.src = item.thumb_rel;
        img.alt = item.sample_key;
        button.appendChild(img);
        strip.appendChild(button);
      }}
    }}
    function render() {{
      if (!filtered.length) return;
      const item = filtered[index];
      image.src = item.debug_rel;
      image.alt = item.sample_key;
      document.getElementById('counter').textContent = `${{index + 1}} / ${{filtered.length}}`;
      document.getElementById('modeText').textContent = item.mode;
      document.getElementById('sample').textContent = item.sample_key;
      document.getElementById('trueOkta').textContent = item.true_okta;
      document.getElementById('predOkta').textContent = item.pred_okta;
      document.getElementById('error').textContent = item.abs_okta_error;
      document.getElementById('cloud').textContent = `${{(item.cloud_fraction * 100).toFixed(1)}}%`;
      document.getElementById('threshold').textContent = Number(item.threshold).toFixed(3);
      document.getElementById('sigma').textContent = Number(item.sigma).toFixed(4);
      document.getElementById('debugLink').href = item.debug_rel;
      document.getElementById('overlayLink').href = item.overlay_rel;
      document.getElementById('maskLink').href = item.mask_rel;
      renderStrip();
    }}
    document.getElementById('prev').onclick = () => setIndex(index - 1);
    document.getElementById('next').onclick = () => setIndex(index + 1);
    mode.onchange = applyFilter;
    okta.onchange = applyFilter;
    sort.onchange = applyFilter;
    window.addEventListener('keydown', event => {{
      if (event.key === 'ArrowLeft') setIndex(index - 1);
      if (event.key === 'ArrowRight') setIndex(index + 1);
    }});
    applyFilter();
  </script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test standard HYTA and low_okta_guard on sorted IMX385 data.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", choices=["day", "night"], default="day")
    parser.add_argument("--only-oktas", type=parse_oktas, default=None)
    parser.add_argument("--modes", type=parse_modes, default=parse_modes("hyta,low_okta_guard"))
    parser.add_argument("--source", choices=["raw", "preview"], default="raw")
    parser.add_argument("--image-name", default="captured_preview.png")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--bayer-pattern", choices=["BGGR", "RGGB", "GRBG", "GBRG"], default="BGGR")
    parser.add_argument("--hyta-tf", type=float, default=0.15)
    parser.add_argument("--hyta-sigma-thr", type=float, default=0.01)
    parser.add_argument("--levels", type=int, default=256)
    parser.add_argument("--threshold-radius-scale", type=float, default=0.55)
    parser.add_argument("--prediction-radius-scale", type=float, default=0.65)
    parser.add_argument("--bottom-boost", type=float, default=0.125)
    parser.add_argument("--bottom-boost-start", type=float, default=0.55)
    parser.add_argument("--bottom-boost-gate", type=float, default=0.35)
    parser.add_argument("--low-okta-base-cf-max", type=float, default=0.25)
    parser.add_argument("--low-okta-cap-okta", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hyta = load_export_capture_hyta()
    samples = list(iter_samples(args.dataset_root, args.split))
    if args.only_oktas is not None:
        samples = [sample for sample in samples if int(sample["true_okta"]) in args.only_oktas]
    if args.max_images is not None:
        samples = samples[: args.max_images]
    if not samples:
        raise RuntimeError(f"No samples found below {args.dataset_root / args.split}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, sample in enumerate(samples, start=1):
        bgr = load_preview_bgr(hyta, sample, args)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        prediction_mask, threshold_mask = valid_masks(hyta, bgr.shape[:2], args)
        base_mask, base_thr, base_sigma, lamn = predict_hyta(hyta, rgb, prediction_mask, threshold_mask, args)

        for mode in args.modes:
            if mode == "hyta":
                mask, threshold, sigma = base_mask, base_thr, base_sigma
            elif mode == "low_okta_guard":
                mask, threshold, sigma = predict_low_okta_guard(base_mask, base_thr, base_sigma, lamn, prediction_mask, args)
            else:
                raise RuntimeError(f"Unhandled mode: {mode}")

            cloud_fraction = float(mask[prediction_mask].mean()) if np.any(prediction_mask) else 0.0
            pred_okta = int(np.clip(round(cloud_fraction * 8.0), 0, 8))
            out_dir = args.output_dir / mode / args.split / f"{sample['true_okta']}_oktas" / sample["sample_dir"].name
            out_prefix = out_dir / "variant"
            label = (
                f"{sample['sample_key']}  mode={mode}  true={sample['true_okta']}  "
                f"pred={pred_okta}  cloud={cloud_fraction:.1%}  thr={threshold:.3f}"
            )
            draw_output(bgr, mask, prediction_mask, label, out_prefix)
            rows.append(
                {
                    "mode": mode,
                    "sample_key": sample["sample_key"],
                    "split": args.split,
                    "true_okta": sample["true_okta"],
                    "pred_okta": pred_okta,
                    "abs_okta_error": abs(pred_okta - sample["true_okta"]),
                    "cloud_fraction": cloud_fraction,
                    "threshold": float(threshold),
                    "sigma": float(sigma),
                    "sample_dir": str(sample["sample_dir"].resolve()),
                    "debug_rel": rel(out_prefix.with_name("variant_debug.png"), args.output_dir),
                    "thumb_rel": rel(out_prefix.with_name("variant_thumb.jpg"), args.output_dir),
                    "overlay_rel": rel(out_prefix.with_name("variant_overlay.png"), args.output_dir),
                    "mask_rel": rel(out_prefix.with_name("variant_mask.png"), args.output_dir),
                }
            )

        if idx == 1 or idx % 20 == 0 or idx == len(samples):
            print(f"Processed {idx}/{len(samples)} images")

    summary_rows = summarize_by_mode(rows)
    by_okta_rows = summarize_by_mode_and_okta(rows)
    confusion = confusion_rows(rows)
    write_csv(args.output_dir / "variant_predictions.csv", rows)
    write_csv(args.output_dir / "variant_summary.csv", summary_rows)
    write_csv(args.output_dir / "variant_by_okta.csv", by_okta_rows)
    write_csv(args.output_dir / "variant_confusion_matrix.csv", confusion)
    write_index(args.output_dir / "index.html", rows)
    print_summary_table(summary_rows)
    print(f"\nSaved variant viewer to: {(args.output_dir / 'index.html').resolve()}")
    print(f"Saved predictions to: {(args.output_dir / 'variant_predictions.csv').resolve()}")
    print(f"Saved summary to: {(args.output_dir / 'variant_summary.csv').resolve()}")
    print(f"Saved per-okta summary to: {(args.output_dir / 'variant_by_okta.csv').resolve()}")
    print(f"Saved confusion matrix to: {(args.output_dir / 'variant_confusion_matrix.csv').resolve()}")


if __name__ == "__main__":
    main()
