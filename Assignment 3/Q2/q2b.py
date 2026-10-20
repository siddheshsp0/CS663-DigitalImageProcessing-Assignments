import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def bilateral_filter(image, sigma_s, sigma_r, strip_rows=64):
    # BF[I]_p = (1 / W_p) * sum_{q in S} G_sigma_s(||p - q||) G_sigma_r(|I_p - I_q|) I_q
    # W_p     =             sum_{q in S} G_sigma_s(||p - q||) G_sigma_r(|I_p - I_q|)
    # S = disc of radius 3*sigma_s around p (spatial Gaussian ~ 0 beyond it).
    # For RGB images, |I_p - I_q| is the Euclidean colour distance so that
    # all 3 channels get the same weights.
    image = image.astype(np.float32)
    h, w = image.shape[:2]

    r = int(np.ceil(3 * sigma_s))

    offsets = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy * dy + dx * dx <= r * r:
                g_s = np.exp(-(dy * dy + dx * dx) / (2 * sigma_s ** 2))
                offsets.append((dy, dx, np.float32(g_s)))

    padded = np.pad(image, ((r, r), (r, r), (0, 0)), mode="reflect")
    output = np.zeros_like(image)

    def filter_strip(y0):
        y1 = min(y0 + strip_rows, h)

        center = image[y0:y1]
        numerator = np.zeros_like(center)
        W = np.zeros(center.shape[:2], dtype=np.float32)

        for dy, dx, g_s in offsets:

            neighbour = padded[y0 + r + dy:y1 + r + dy, r + dx:r + dx + w]

            dist_sq = np.sum((neighbour - center) ** 2, axis=2)
            weight = g_s * np.exp(-dist_sq / (2 * sigma_r ** 2))

            numerator += weight[:, :, None] * neighbour
            W += weight

        output[y0:y1] = numerator / W[:, :, None]

    # Strips of rows are independent -> filter them in parallel threads
    # (keeps memory usage low; numpy releases the GIL for array operations)
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        list(pool.map(filter_strip, range(0, h, strip_rows)))

    return output


def load_image(path):
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Work with floating point intensities in [0, 1]
    return img.astype(np.float32) / 255.0


data_dir = Path(__file__).resolve().parent.parent / "data" / "diffusion"
output_dir = Path(__file__).resolve().parent.parent / "output"

image_names = [
    "colorCheckerNoisy.png",
    "rajaRaviVarma.png",
    "sagarmaathaa.png",
    "racing.png",
]

# Tuned parameters (sigma_s in pixels, sigma_r in [0, 1] intensity units)
bf_params = {
    "colorCheckerNoisy.png": dict(sigma_s=3.0, sigma_r=0.35),
    "rajaRaviVarma.png":     dict(sigma_s=3.0, sigma_r=0.18),
    "sagarmaathaa.png":      dict(sigma_s=3.0, sigma_r=0.12),
    "racing.png":            dict(sigma_s=3.0, sigma_r=0.25),
}


if __name__ == "__main__":

    output_dir.mkdir(exist_ok=True)

    for idx, name in enumerate(image_names, start=1):

        img = load_image(data_dir / name)
        params = bf_params[name]

        denoised = bilateral_filter(img, **params)

        h, w = img.shape[:2]
        fig, ax = plt.subplots(1, 2, figsize=(16, 8 * h / w + 0.8))

        ax[0].imshow(img)
        ax[0].set_title("Noisy input")

        ax[1].imshow(np.clip(denoised, 0, 1))
        ax[1].set_title(
            f"Bilateral (sigma_s={params['sigma_s']}, "
            f"sigma_r={params['sigma_r']})"
        )

        for a in ax:
            a.axis("off")

        plt.suptitle(name)
        plt.tight_layout()
        plt.savefig(output_dir / f"q2b{idx}.png", dpi=150, bbox_inches="tight")
        plt.show()
