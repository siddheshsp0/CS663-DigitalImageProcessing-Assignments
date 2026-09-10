import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_THRESHOLDS = {
    "data/thresh/blackboard.png": 55,
    "data/thresh/qr.png": 135,
    "data/thresh/receipt.png": 150,
}


def myManualThreshold(img: np.ndarray, threshold: float):

    binary = np.zeros_like(img, dtype=np.uint8)

    binary[img >= threshold] = 255

    return binary

if __name__ == "__main__":

    fig, axes = plt.subplots(
        len(IMAGE_THRESHOLDS), 2,
        figsize=(6, 2.5 * len(IMAGE_THRESHOLDS))
    )

    for row, (input_image, threshold) in enumerate(
        IMAGE_THRESHOLDS.items()
    ):
        img = cv2.imread(
            input_image,
            cv2.IMREAD_GRAYSCALE
        )

        if img is None:
            raise FileNotFoundError(
                f"Could not read {input_image}"
            )

        binary = myManualThreshold(
            img,
            threshold
        )

        axes[row, 0].imshow(
            img,
            cmap="gray",
            aspect="equal"
        )

        axes[row, 0].set_title(
            input_image.split("/")[-1] + " - Original"
        )
        axes[row, 0].set_xlabel("Column")
        axes[row, 0].set_ylabel("Row")

        axes[row, 1].imshow(
            binary,
            cmap="gray",
            vmin=0,
            vmax=255,
            aspect="equal"
        )

        axes[row, 1].set_title(
            f"Manual Thresholding (T = {threshold})"
        )

        axes[row, 1].set_xlabel("Column")
        axes[row, 1].set_ylabel("Row")

    plt.tight_layout()
    plt.show()