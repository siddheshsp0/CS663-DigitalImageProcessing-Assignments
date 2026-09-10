import cv2
import numpy as np
import matplotlib.pyplot as plt


SOURCE_PATH = "data/hist/retina.png"
REFERENCE_PATH = "data/hist/retinaRef.png"


def myHistMatch(source, reference, bins=64):

    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)

    matched_lab = source_lab.copy()

    def foreground_mask(img, thresh=8):
        return np.any(img > thresh, axis=2)

    source_mask = foreground_mask(source)
    reference_mask = foreground_mask(reference)

    def match_channel(src_channel, ref_channel,
                      src_mask, ref_mask):

        src_values = src_channel[src_mask].astype(np.float64)
        ref_values = ref_channel[ref_mask].astype(np.float64)

        # Histogram
        src_hist, bin_edges = np.histogram(
            src_values,
            bins=bins,
            range=(0, 256)
        )

        ref_hist, _ = np.histogram(
            ref_values,
            bins=bins,
            range=(0, 256)
        )

        # CDF
        src_cdf = np.cumsum(src_hist).astype(np.float64)
        ref_cdf = np.cumsum(ref_hist).astype(np.float64)

        src_cdf /= src_cdf[-1]
        ref_cdf /= ref_cdf[-1]

        # Mapping:
        # source CDF and reference CDF
        mapping = np.zeros(bins)

        for i in range(bins):
            # Find reference bin whose CDF is closest
            j = np.argmin(np.abs(ref_cdf - src_cdf[i]))
            mapping[i] = j

        # Convert src pixel value to src bin
        src_bin = np.floor(
            src_values / 256.0 * bins
        ).astype(int)

        src_bin = np.clip(src_bin, 0, bins - 1)

        # Convert mapped bin back to intensity
        matched_values = (
            (mapping[src_bin] + 0.5)
            * 256.0 / bins
        )

        matched_values = np.clip(
            matched_values,
            0,
            255
        )

        return matched_values.astype(np.uint8)

    # Match L, a and b independently
    for channel in range(3):

        matched_values = match_channel(
            source_lab[:, :, channel],
            reference_lab[:, :, channel],
            source_mask,
            reference_mask
        )

        matched_lab[:, :, channel][source_mask] = matched_values

    matched = cv2.cvtColor(
        matched_lab,
        cv2.COLOR_LAB2BGR
    )

    # Keep background black
    matched[~source_mask] = 0

    return matched

def plot_histogram(img, title, bins=64):

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    mask = np.any(img != 0, axis=2)

    labels = ["L", "a", "b"]

    plt.figure(figsize=(12, 4))

    for i in range(3):

        values = lab[:, :, i][mask]

        plt.subplot(1, 3, i + 1)

        plt.hist(
            values,
            bins=bins,
            range=(0, 256)
        )

        plt.title(f"{title} - {labels[i]}")
        plt.xlabel("Intensity")
        plt.ylabel("Frequency")

    plt.tight_layout()


if __name__ == "__main__":

    # Load images
    source = cv2.imread(SOURCE_PATH)
    reference = cv2.imread(REFERENCE_PATH)

    if source is None:
        raise FileNotFoundError(SOURCE_PATH)

    if reference is None:
        raise FileNotFoundError(REFERENCE_PATH)

    # Try different bin sizes
    bin_settings = [8, 32, 64, 128, 256]

    for bins in bin_settings:

        matched = myHistMatch(
            source,
            reference,
            bins=bins
        )

        # Images side by side
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(reference, cv2.COLOR_BGR2RGB))
        plt.title("Reference")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(matched, cv2.COLOR_BGR2RGB))
        plt.title(f"Matched ({bins} bins)")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

        plot_histogram(
            source,
            f"Original ({bins} bins)",
            bins
        )

        plot_histogram(
            reference,
            f"Reference ({bins} bins)",
            bins
        )

        plot_histogram(
            matched,
            f"Matched ({bins} bins)",
            bins
        )

        plt.show()