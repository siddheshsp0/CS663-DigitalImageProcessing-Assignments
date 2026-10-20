import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def diffusion_coefficient(grad_mag, K, option):
    # Two choices of c(||grad I||) proposed by Perona-Malik
    if option == "exp":
        return np.exp(-(grad_mag / K) ** 2)

    return 1.0 / (1.0 + (grad_mag / K) ** 2)


def perona_malik(image, num_iter, K, lam, option="exp"):
    # I_new = I + lam * (cN dI_N + cS dI_S + cE dI_E + cW dI_W)
    # with dI_N = I(north neighbour) - I, cN = g(|dI_N|), etc.
    # For RGB images, |dI| is the colour-difference norm across channels so
    # that all 3 channels share the same coefficient and edges stay aligned.
    I = image.astype(np.float32).copy()

    for t in range(num_iter):

        # dI_S at row i and dI_N at row i+1 are the same difference with
        # opposite sign (and the same c), so compute each pair only once.
        dS = I[1:] - I[:-1]
        dE = I[:, 1:] - I[:, :-1]

        cS = diffusion_coefficient(np.sqrt(np.sum(dS ** 2, axis=2)), K, option)
        cE = diffusion_coefficient(np.sqrt(np.sum(dE ** 2, axis=2)), K, option)

        flux_S = lam * cS[:, :, None] * dS
        flux_E = lam * cE[:, :, None] * dE

        # No neighbour outside the image -> zero flux across the border
        I[:-1] += flux_S       # + lam cS dI_S
        I[1:] -= flux_S        # + lam cN dI_N
        I[:, :-1] += flux_E    # + lam cE dI_E
        I[:, 1:] -= flux_E     # + lam cW dI_W

    return I


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

# Tuned parameters (intensities in [0, 1]); lam <= 0.25 for stability
pm_params = {
    "colorCheckerNoisy.png": dict(num_iter=30, K=0.12, lam=0.2, option="inv"),
    "rajaRaviVarma.png":     dict(num_iter=20, K=0.05, lam=0.2, option="inv"),
    "sagarmaathaa.png":      dict(num_iter=20, K=0.035, lam=0.2, option="inv"),
    "racing.png":            dict(num_iter=30, K=0.07, lam=0.2, option="inv"),
}


if __name__ == "__main__":

    output_dir.mkdir(exist_ok=True)

    for idx, name in enumerate(image_names, start=1):

        img = load_image(data_dir / name)
        params = pm_params[name]

        denoised = perona_malik(img, **params)

        h, w = img.shape[:2]
        fig, ax = plt.subplots(1, 2, figsize=(16, 8 * h / w + 0.8))

        ax[0].imshow(img)
        ax[0].set_title("Noisy input")

        ax[1].imshow(np.clip(denoised, 0, 1))
        ax[1].set_title(
            f"Perona-Malik (iter={params['num_iter']}, K={params['K']}, "
            f"lambda={params['lam']}, c={params['option']})"
        )

        for a in ax:
            a.axis("off")

        plt.suptitle(name)
        plt.tight_layout()
        plt.savefig(output_dir / f"q2a{idx}.png", dpi=150, bbox_inches="tight")
        plt.show()
