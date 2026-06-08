import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

IMG_SIZE = 304
MODEL_PATH = "tiny_ucloudnet_pc.h5"
IMAGE_PATH = "data/images/0055.png"


def load_image(image_path, img_size=304):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Fant ikke bilde: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original = img.copy()

    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # (1, H, W, C)

    return original, img


def main():
    model = tf.keras.models.load_model(MODEL_PATH)

    original, input_img = load_image(IMAGE_PATH, IMG_SIZE)

    pred = model.predict(input_img)[0]          # (304, 304, 1)
    pred_mask = (pred > 0.5).astype(np.uint8)   # terskel
    pred_mask = pred_mask.squeeze()             # (304, 304)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(pred.squeeze(), cmap="gray")
    plt.title("Sannsynlighet")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask, cmap="gray")
    plt.title("Maske")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()