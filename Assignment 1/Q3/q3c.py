import cv2
import numpy as np
import matplotlib.pyplot as plt

#params
NUM_BINS = 64
# Main tuned neighborhood size
WINDOW_SIZE = 101
# Histogram clipping threshold in [0, 1]
# Clip limit = CLIP_THRESHOLD * number of pixels in window
CLIP_THRESHOLD = 0.05
# Automatically used for the half-threshold experiment
HALF_THRESHOLD = CLIP_THRESHOLD / 2.0
# Significantly larger/smaller windows
LARGE_WINDOW_SIZE = 401
SMALL_WINDOW_SIZE = 21

def myCLAHE(img: np.ndarray,num_bins: int,window_size: int,clip_threshold: float):

    if not (0.0 <= clip_threshold <= 1.0):
        raise ValueError("clip_threshold must be in [0, 1]")

    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")

    # RGB -> YCrCb
    rgb = img.astype(np.float32) / 255.0
    ycrcb = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2YCrCb
    )

    # Luminance in [0,1]
    Y = ycrcb[:, :, 0].astype(np.float64)

    M, N = Y.shape
    equalized_Y = np.zeros_like(Y)

    # Half window size
    radius = window_size // 2

    #Process every pixel
    for i in range(M):
        if i % 50 == 0:
            print(
                f"Processing row {i}/{M} "
                f"({100*i/M:.1f}%)"
            )

        # Window boundaries
        # np.clip effectively crops the window at boundaries
        r0 = max(0, i - radius)
        r1 = min(M, i + radius + 1)

        for j in range(N):

            c0 = max(0, j - radius)
            c1 = min(N, j + radius + 1)
            # Local window
            local = Y[r0:r1, c0:c1]

            # Local histogram
            hist, _ = np.histogram(
                local,
                bins=num_bins,
                range=(0.0, 1.0)
            )

            hist = hist.astype(np.float64)

            # CLIP HISTOGRAM
            # Number of pixels actually inside the window
            window_pixels = local.size

            # Threshold is normalized by window size
            clip_limit = clip_threshold * window_pixels

            # Amount clipped from each bin
            excess = np.sum(
                np.maximum(hist - clip_limit, 0)
            )

            # Clip histogram
            hist = np.minimum(
                hist,
                clip_limit
            )

            # redistributing histogram mass uniformly
            hist += excess / num_bins
            # CDF
            cdf = np.cumsum(hist)
            # Normalize CDF to [0,1]
            cdf = cdf / cdf[-1]

            # Map center pixel through CDF
            center_value = Y[i, j]
            bin_index = int(
                np.floor(center_value * num_bins)
            )
            bin_index = np.clip(bin_index,0,num_bins - 1)
            equalized_Y[i, j] = cdf[bin_index]

    print("Processing complete.")

    # Replace luminance
    ycrcb_equalized = ycrcb.copy()

    ycrcb_equalized[:, :, 0] = (
        equalized_Y.astype(np.float32)
    )

    # YCrCb -> RGB
    equalized_rgb = cv2.cvtColor(
        ycrcb_equalized,
        cv2.COLOR_YCrCb2RGB
    )

    equalized_rgb = np.clip(
        equalized_rgb,
        0.0,
        1.0
    )

    # Convert back to uint8 for display
    equalized_rgb = (
        equalized_rgb * 255
    ).astype(np.uint8)

    return equalized_rgb, Y, equalized_Y


def plot_histogram(ax, Y, title, num_bins):
    ax.hist(Y.ravel(),bins=num_bins,range=(0, 1))
    ax.set_title(title)
    ax.set_xlabel("Luminance")
    ax.set_ylabel("Number of Pixels")
    ax.set_xlim(0, 1)


#display results for canyon
def show_canyon_results(
    original_rgb,
    original_Y,
    tuned,
    large,
    small,
    half
):

    tuned_rgb, tuned_Y = tuned
    large_rgb, large_Y = large
    small_rgb, small_Y = small
    half_rgb, half_Y = half

    # Images
    fig, axes = plt.subplots(
        1, 5,
        figsize=(22, 5)
    )

    axes[0].imshow(original_rgb)
    axes[0].set_title("Original")

    axes[1].imshow(tuned_rgb)
    axes[1].set_title(
        f"Tuned\n"
        f"Bins={NUM_BINS}, "
        f"W={WINDOW_SIZE}, "
        f"T={CLIP_THRESHOLD}"
    )

    axes[2].imshow(large_rgb)
    axes[2].set_title(
        f"Large Window\n"
        f"W={LARGE_WINDOW_SIZE}"
    )

    axes[3].imshow(small_rgb)
    axes[3].set_title(
        f"Small Window\n"
        f"W={SMALL_WINDOW_SIZE}"
    )

    axes[4].imshow(half_rgb)
    axes[4].set_title(
        f"Half Threshold\n"
        f"T={HALF_THRESHOLD}"
    )

    for ax in axes:
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

    fig.suptitle(
        "Canyon - CLAHE Results",
        fontsize=16
    )

    plt.tight_layout()
    plt.show()

    # Histogram
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))

    plot_histogram(axes[0],original_Y,"Original Histogram",NUM_BINS)
    plot_histogram(axes[1],tuned_Y,f"Tuned\n"f"Bins={NUM_BINS}, W={WINDOW_SIZE}, "f"T={CLIP_THRESHOLD}",NUM_BINS)
    plot_histogram(axes[2],large_Y,f"Large Window\nW={LARGE_WINDOW_SIZE}",NUM_BINS)
    plot_histogram(axes[3],small_Y,f"Small Window\nW={SMALL_WINDOW_SIZE}",NUM_BINS)
    plot_histogram(axes[4],half_Y,f"Half Threshold\nT={HALF_THRESHOLD}",NUM_BINS)
    fig.suptitle("Canyon - Luminance Histograms",fontsize=16)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    canyon_bgr = cv2.imread(
        "data/hist/canyon.png"
    )
    if canyon_bgr is None:
        raise FileNotFoundError(
            "Could not read canyon.png"
        )
    canyon_rgb = cv2.cvtColor(
        canyon_bgr,
        cv2.COLOR_BGR2RGB
    )

    # Get original luminance
    canyon_float = (
        canyon_rgb.astype(np.float32) / 255.0
    )

    canyon_ycrcb = cv2.cvtColor(
        canyon_float,
        cv2.COLOR_RGB2YCrCb
    )

    canyon_Y = canyon_ycrcb[:, :, 0]
    print("\n========== TUNED CLAHE ==========")

    tuned_rgb, _, tuned_Y = myCLAHE(
        canyon_rgb,
        NUM_BINS,
        WINDOW_SIZE,
        CLIP_THRESHOLD
    )

    print("\n========== LARGE WINDOW ==========")

    large_rgb, _, large_Y = myCLAHE(
        canyon_rgb,
        NUM_BINS,
        LARGE_WINDOW_SIZE,
        CLIP_THRESHOLD
    )
    print("\n========== SMALL WINDOW ==========")

    small_rgb, _, small_Y = myCLAHE(
        canyon_rgb,
        NUM_BINS,
        SMALL_WINDOW_SIZE,
        CLIP_THRESHOLD
    )


    print("\n========== HALF THRESHOLD ==========")

    half_rgb, _, half_Y = myCLAHE(
        canyon_rgb,
        NUM_BINS,
        WINDOW_SIZE,
        HALF_THRESHOLD
    )
    show_canyon_results(
        canyon_rgb,
        canyon_Y,
        (tuned_rgb, tuned_Y),
        (large_rgb, large_Y),
        (small_rgb, small_Y),
        (half_rgb, half_Y)
    )