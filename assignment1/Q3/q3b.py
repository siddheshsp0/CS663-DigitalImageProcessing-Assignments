import cv2
import numpy as np
import matplotlib.pyplot as plt


IMAGE_PATH = "data/hist/leh.png"

def myHistEqualize(img: np.ndarray):
    # Convert BGR uint8 -> YCrCb float32
    img_float = img.astype(np.float32) / 255.0

    ycrcb = cv2.cvtColor(img_float,cv2.COLOR_BGR2YCrCb)

    # Extract luminance
    # Y is in [0, 1]
    Y = ycrcb[:, :, 0]

    # Histogram of luminance
    num_bins = 256

    histogram, _ = np.histogram(Y.ravel(), bins=num_bins, range=(0.0, 1.0))

    # Cumulative distribution function
    cdf = np.cumsum(histogram)

    # First non-zero histogram bin
    nonzero = np.nonzero(histogram)[0]

    if len(nonzero) == 0:
        return img_float, Y, Y

    cdf_min = cdf[nonzero[0]]

    # Normalize CDF
    cdf_normalized = ((cdf - cdf_min) / (Y.size - cdf_min))

    cdf_normalized = np.clip(cdf_normalized, 0.0, 1.0)

    # map each pixel through CDF

    bin_indices = np.floor(Y * num_bins).astype(int)

    bin_indices = np.clip(bin_indices,0,num_bins - 1)

    Y_equalized = cdf_normalized[bin_indices]

    # Replace luminance, keep chrominance unchanged (colors unchanged)
    ycrcb_equalized = ycrcb.copy()
    ycrcb_equalized[:, :, 0] = Y_equalized

    # YCrCb -> BGR

    equalized_bgr = cv2.cvtColor(ycrcb_equalized,cv2.COLOR_YCrCb2BGR)
    equalized_bgr = np.clip(equalized_bgr,0.0,1.0)
    return equalized_bgr, Y, Y_equalized


if __name__ == "__main__":
    img_bgr = cv2.imread(IMAGE_PATH)
    if img_bgr is None:
        raise FileNotFoundError(
            f"Could not read image: {IMAGE_PATH}"
        )

    img_rgb = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2RGB
    )

    # Histogram equalization
    stretched_rgb, original_Y, equalized_Y = myHistEqualize(img_rgb)


    # Display images and luminance histograms
    fig, axes = plt.subplots(
        2, 2,
        figsize=(14, 10)
    )

    # Original image
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].set_xlabel("Column")
    axes[0, 0].set_ylabel("Row")

    # Histogram-equalized image
    axes[0, 1].imshow(stretched_rgb)
    axes[0, 1].set_title(
        "Nonlinear Contrast Stretched Image"
    )
    axes[0, 1].set_xlabel("Column")
    axes[0, 1].set_ylabel("Row")

    # Original luminance histogram
    axes[1, 0].hist(
        original_Y.ravel(),
        bins=256,
        range=(0, 1)
    )
    axes[1, 0].set_title(
        "Histogram of Original Luminance"
    )
    axes[1, 0].set_xlabel("Luminance")
    axes[1, 0].set_ylabel("Number of Pixels")
    axes[1, 0].set_xlim(0, 1)

    # Equalized luminance histogram
    axes[1, 1].hist(
        equalized_Y.ravel(),
        bins=256,
        range=(0, 1)
    )
    axes[1, 1].set_title(
        "Histogram of Contrast-Enhanced Luminance"
    )
    axes[1, 1].set_xlabel("Luminance")
    axes[1, 1].set_ylabel("Number of Pixels")
    axes[1, 1].set_xlim(0, 1)

    plt.tight_layout()
    plt.show()