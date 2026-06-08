from __future__ import annotations

import argparse
import csv
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Model, layers


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT_CSV = REPO_ROOT / "splits" / "swinyseg" / "swinyseg_split_both.csv"
DEFAULT_MODEL_OUT = REPO_ROOT / "SW" / "tiny-ucloudnet" / "models" / "tiny_ucloudnet_swinyseg.keras"
DEFAULT_HISTORY_OUT = REPO_ROOT / "SW" / "tiny-ucloudnet" / "models" / "tiny_ucloudnet_swinyseg_history.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Tiny U Cloud Net on the SWINySeg train split. By default, "
            "training uses both day/night and original/augmented images."
        )
    )
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUT)
    parser.add_argument("--history-out", type=Path, default=DEFAULT_HISTORY_OUT)
    parser.add_argument("--subset", choices=["both", "day", "night"], default="both")
    parser.add_argument("--img-size", type=int, default=304)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    return parser.parse_args()


def load_rows(
    split_csv: Path,
    split: str,
    subset: str,
    originals_only: bool,
) -> list[dict[str, str]]:
    if not split_csv.exists():
        raise FileNotFoundError(f"Fant ikke split-fil: {split_csv}")

    rows = []
    with split_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] != split:
                continue
            if subset != "both" and row["time_of_day"] != subset:
                continue
            if originals_only and row["is_original"].lower() != "true":
                continue
            rows.append(row)

    if not rows:
        raise RuntimeError(
            f"Ingen rader funnet for split={split}, subset={subset}, "
            f"originals_only={originals_only}"
        )
    return rows


def print_rows_stats(label: str, rows: list[dict[str, str]]) -> None:
    n_day = sum(1 for row in rows if row["time_of_day"] == "day")
    n_night = sum(1 for row in rows if row["time_of_day"] == "night")
    n_original = sum(1 for row in rows if row["is_original"].lower() == "true")
    n_augmented = len(rows) - n_original
    print(
        f"{label}: {len(rows)} images "
        f"(original={n_original}, augmented={n_augmented}, day={n_day}, night={n_night})"
    )


def decode_sample(image_path: tf.Tensor, mask_path: tf.Tensor, img_size: int):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_size, img_size], method="bilinear")
    image = tf.cast(image, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, [img_size, img_size], method="nearest")
    mask = tf.cast(mask > 0, tf.float32)

    return image, mask


def make_dataset(
    rows: list[dict[str, str]],
    img_size: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    image_paths = [row["image_path"] for row in rows]
    mask_paths = [row["mask_path"] for row in rows]

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(rows), seed=seed, reshuffle_each_iteration=True)

    dataset = dataset.map(
        lambda image_path, mask_path: decode_sample(image_path, mask_path, img_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def tiny_ucloudnet(input_shape=(304, 304, 3)) -> Model:
    inputs = layers.Input(shape=input_shape)

    e1 = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    p1 = layers.MaxPooling2D(pool_size=(2, 2))(e1)

    e2 = layers.SeparableConv2D(8, 3, padding="same", activation="relu")(p1)
    p2 = layers.MaxPooling2D(pool_size=(2, 2))(e2)

    e3 = layers.SeparableConv2D(8, 3, padding="same", activation="relu")(p2)
    p3 = layers.MaxPooling2D(pool_size=(2, 2))(e3)

    d1 = layers.UpSampling2D(size=(2, 2))(p3)
    d1 = layers.Concatenate()([d1, e3])
    d1 = layers.SeparableConv2D(8, 3, padding="same", activation="relu")(d1)

    d2 = layers.UpSampling2D(size=(2, 2))(d1)
    d2 = layers.Concatenate()([d2, e2])
    d2 = layers.SeparableConv2D(8, 3, padding="same", activation="relu")(d2)

    d3 = layers.UpSampling2D(size=(2, 2))(d2)
    d3 = layers.Concatenate()([d3, e1])
    d3 = layers.Conv2D(16, 3, padding="same", activation="relu")(d3)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(d3)
    return Model(inputs, outputs)


def dice_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)

    return (2.0 * intersection + 1e-7) / (denominator + 1e-7)


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    train_rows = load_rows(args.split_csv, split="train", subset=args.subset, originals_only=False)
    val_rows = load_rows(args.split_csv, split="val", subset=args.subset, originals_only=True)

    print(f"Split CSV: {args.split_csv.resolve()}")
    print_rows_stats("Train", train_rows)
    print_rows_stats("Validation originals", val_rows)

    train_ds = make_dataset(
        train_rows,
        img_size=args.img_size,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    val_ds = make_dataset(
        val_rows,
        img_size=args.img_size,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )

    model = tiny_ucloudnet((args.img_size, args.img_size, 3))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", dice_metric],
    )
    model.summary()

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.history_out.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.early_stopping_patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(args.model_out),
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.CSVLogger(str(args.history_out)),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    model.save(args.model_out)
    print(f"Lagret modell til: {args.model_out.resolve()}")
    print(f"Lagret treningshistorikk til: {args.history_out.resolve()}")


if __name__ == "__main__":
    main()
