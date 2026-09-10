import cv2
import numpy as np
import matplotlib.pyplot as plt


INPUT_IMAGES = [
    "data/thresh/blackboard.png",
    "data/thresh/qr.png",
    "data/thresh/receipt.png",
]

# Must be odd
BLOCK_SIZE = [531, 151, 301]
# Sauvola parameter
K = [0.2, 0.2, 0.15]


def mySauvolaThreshold(img, block_size, k):

    if block_size % 2 == 0:
        raise ValueError(
            "block_size must be odd"
        )

    img_float = img.astype(np.float64)

    local_mean = cv2.boxFilter(
        img_float,
        ddepth=-1,
        ksize=(block_size, block_size),
        normalize=True,
        borderType=cv2.BORDER_CONSTANT
    )

    # Local variance
    local_mean_square = cv2.boxFilter(
        img_float ** 2,
        ddepth=-1,
        ksize=(block_size, block_size),
        normalize=True,
        borderType=cv2.BORDER_CONSTANT
    )

    local_variance = (
        local_mean_square - local_mean ** 2
    )

    local_variance = np.maximum(
        local_variance,
        0
    )

    # Local standard deviation
    local_std = np.sqrt(local_variance)

    # R = maximum local standard deviation
    R = np.max(local_std)

    # Sauvola threshold
    threshold_image = (
        local_mean
        + local_mean * k * (local_std / R - 1)
    )

    binary = np.zeros_like(
        img,
        dtype=np.uint8
    )

    binary[img >= threshold_image] = 255

    return binary, threshold_image


if __name__ == "__main__":

    fig, axes = plt.subplots(
        len(INPUT_IMAGES), 3,
        figsize=(9, 3 * len(INPUT_IMAGES))
    )

    for row, input_image in enumerate(INPUT_IMAGES):
        img = cv2.imread(
            input_image,
            cv2.IMREAD_GRAYSCALE
        )

        if img is None:
            raise FileNotFoundError(
                f"Could not read {input_image}"
            )

        binary, threshold_image = mySauvolaThreshold(
            img,
            BLOCK_SIZE[row],
            K[row]
        )

        image_name = input_image.split("/")[-1]

        axes[row, 0].imshow(
            img,
            cmap="gray",
            aspect="equal"
        )

        axes[row, 1].set_title(
            f"Sauvola Thresholding\n"
            f"Block = {BLOCK_SIZE[row]}, k = {K[row]}"
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
            f"Sauvola Thresholding\n"
            f"Block = {BLOCK_SIZE[row]}, k = {K[row]}"
        )
        axes[row, 1].set_xlabel("Column")
        axes[row, 1].set_ylabel("Row")

        im = axes[row, 2].imshow(
            threshold_image,
            cmap="jet",
            aspect="equal"
        )

        axes[row, 2].set_title(
            "Per-Pixel Sauvola Threshold"
        )
        axes[row, 2].set_xlabel("Column")
        axes[row, 2].set_ylabel("Row")

        fig.colorbar(
            im,
            ax=axes[row, 2],
            label="Threshold"
        )

    plt.tight_layout()
    plt.show()