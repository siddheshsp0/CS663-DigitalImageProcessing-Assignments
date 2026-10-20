import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from q2a import perona_malik, pm_params, load_image, data_dir, output_dir, image_names
from q2b import bilateral_filter, bf_params


# Zoom regions (y, x, height, width) in pixel coordinates of each image
zoom_regions = {
    "colorCheckerNoisy.png": [(260, 60, 150, 270), (60, 268, 150, 270)],
    "rajaRaviVarma.png":     [(380, 260, 200, 300), (150, 60, 200, 300)],
    "sagarmaathaa.png":      [(910, 1220, 260, 390), (100, 950, 260, 390)],
    "racing.png":            [(1150, 1450, 300, 450), (640, 2300, 360, 540)],
}

zoom_colors = ["red", "cyan"]


def show_comparison(name, images, titles, regions):
    h, w = images[0].shape[:2]

    # Row heights proportional to the aspect ratio of what is shown in them,
    # so that the images fill the page with very little whitespace
    heights = [h / w] + [rh / rw for (_, _, rh, rw) in regions]

    fig_width = 24
    fig_height = fig_width / len(images) * sum(heights) * 1.05 + 1.0

    fig, ax = plt.subplots(
        1 + len(regions), len(images),
        figsize=(fig_width, fig_height),
        gridspec_kw=dict(height_ratios=heights, wspace=0.02, hspace=0.04),
    )

    for col, (img, title) in enumerate(zip(images, titles)):

        img = np.clip(img, 0, 1)

        ax[0, col].imshow(img)
        ax[0, col].set_title(title, fontsize=16)
        ax[0, col].axis("off")

        for row, ((y, x, rh, rw), color) in enumerate(zip(regions, zoom_colors), start=1):

            ax[0, col].add_patch(Rectangle(
                (x, y), rw, rh, fill=False, edgecolor=color, linewidth=2
            ))

            ax[row, col].imshow(img[y:y + rh, x:x + rw], interpolation="nearest")
            ax[row, col].set_xticks([])
            ax[row, col].set_yticks([])

            for spine in ax[row, col].spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)

            if col == 0:
                ax[row, col].set_ylabel(f"Zoom {row}", fontsize=14)

    # Margins fixed in inches so tall and wide figures look the same
    fig.suptitle(name, fontsize=18, y=1 - 0.1 / fig_height, va="top")
    fig.subplots_adjust(
        left=0.02, right=0.99, top=1 - 0.85 / fig_height, bottom=0.1 / fig_height
    )

    return fig


if __name__ == "__main__":

    output_dir.mkdir(exist_ok=True)

    for idx, name in enumerate(image_names, start=1):

        img = load_image(data_dir / name)

        pm = pm_params[name]
        bf = bf_params[name]

        denoised_pm = perona_malik(img, **pm)
        denoised_bf = bilateral_filter(img, **bf)

        titles = [
            "Original (noisy input)",
            f"Perona-Malik  (iter={pm['num_iter']}, K={pm['K']}, "
            f"lambda={pm['lam']}, c={pm['option']})",
            f"Bilateral  (sigma_s={bf['sigma_s']}, sigma_r={bf['sigma_r']})",
        ]

        fig = show_comparison(
            name, [img, denoised_pm, denoised_bf], titles, zoom_regions[name]
        )

        plt.savefig(output_dir / f"q2c{idx}.png", dpi=150)
        plt.show()
