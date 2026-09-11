import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d
from pathlib import Path


def ncc_channel(image, template):
    h, w = template.shape
    th, tw = image.shape

    # Cross correlation: sum(I * T)
    cross = correlate2d(image, template, mode="valid")

    ones = np.ones((h, w), dtype=np.float64)

    # Sum and sum of squares of every image patch
    patch_sum = correlate2d(image, ones, mode="valid")
    patch_sq_sum = correlate2d(image * image, ones, mode="valid")

    template_sum = np.sum(template)
    template_sq_sum = np.sum(template * template)

    n = h * w

    numerator = cross - (patch_sum * template_sum) / n

    patch_variance = (
        patch_sq_sum
        - (patch_sum * patch_sum) / n
    )

    template_variance = (
        template_sq_sum
        - (template_sum * template_sum) / n
    )

    denominator = np.sqrt(
        patch_variance * template_variance
    )

    ncc = np.zeros_like(numerator, dtype=np.float64)

    valid = denominator > 1e-12
    ncc[valid] = numerator[valid] / denominator[valid]

    return ncc


def ncc(image, template):
    """
    Compute NCC independently for the three color channels.
    """

    result = []

    for c in range(3):
        result.append(
            ncc_channel(
                image[:, :, c],
                template[:, :, c]
            )
        )

    return result


def find_red_annulus_mask(template):
    """
    Find the region inside the outer edge of the red ring.

    The red pixels are used to estimate the center and radius
    of the ring. The returned mask includes the ring and the
    region enclosed by it.
    """

    r = template[:, :, 0]
    g = template[:, :, 1]
    b = template[:, :, 2]

    red = (
        (r > 0.5)
        & (r > g * 1.3)
        & (r > b * 1.3)
    )

    y, x = np.where(red)

    if len(x) == 0:
        raise ValueError("Could not find the red annulus.")

    center_x = np.mean(x)
    center_y = np.mean(y)

    distances = np.sqrt(
        (x - center_x) ** 2
        + (y - center_y) ** 2
    )

    radius = np.max(distances)

    yy, xx = np.indices(template.shape[:2])

    mask = (
        (xx - center_x) ** 2
        + (yy - center_y) ** 2
        <= radius ** 2
    )

    return mask


def masked_ncc_channel(image, template, mask):
    """
    NCC using only pixels selected by mask.
    """

    template_masked = template * mask

    n = np.sum(mask)

    if n == 0:
        raise ValueError("Mask contains no pixels.")

    # sum(I * T) over the selected pixels
    cross = correlate2d(
        image,
        template_masked,
        mode="valid"
    )

    # Sum of image pixels inside the mask
    patch_sum = correlate2d(
        image,
        mask.astype(np.float64),
        mode="valid"
    )

    # Sum of squared image pixels inside the mask
    patch_sq_sum = correlate2d(
        image * image,
        mask.astype(np.float64),
        mode="valid"
    )

    template_sum = np.sum(template_masked)
    template_sq_sum = np.sum(
        template_masked * template_masked
    )

    numerator = (
        cross
        - patch_sum * template_sum / n
    )

    patch_variance = (
        patch_sq_sum
        - patch_sum * patch_sum / n
    )

    template_variance = (
        template_sq_sum
        - template_sum * template_sum / n
    )

    denominator = np.sqrt(
        patch_variance * template_variance
    )

    result = np.zeros_like(
        numerator,
        dtype=np.float64
    )

    valid = denominator > 1e-12

    result[valid] = (
        numerator[valid]
        / denominator[valid]
    )

    return result


def masked_ncc(image, template, mask):
    """
    Compute masked NCC independently for each color channel
    """

    result = []

    for c in range(3):
        result.append(
            masked_ncc_channel(
                image[:, :, c],
                template[:, :, c],
                mask
            )
        )

    return result


def display_ncc(ncc_images, title):
    fig, axes = plt.subplots(
        1, 3,
        figsize=(15, 5)
    )

    for c in range(3):
        im = axes[c].imshow(
            ncc_images[c],
            cmap="gray",
            vmin=-1,
            vmax=1
        )

        axes[c].set_title(
            ["Red channel", "Green channel", "Blue channel"][c]
        )

        axes[c].axis("off")
        fig.colorbar(im, ax=axes[c])

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def resize_template(template, size):
    return cv2.resize(
        template,
        (size, size),
        interpolation=cv2.INTER_LINEAR
    )


data_dir = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "templateMatch"
)

scene_path = data_dir / "parking.png"
template_path = data_dir / "templateNoPark.png"

scene = cv2.imread(str(scene_path))
template = cv2.imread(str(template_path))

if scene is None:
    raise FileNotFoundError(scene_path)

if template is None:
    raise FileNotFoundError(template_path)

scene = cv2.cvtColor(scene, cv2.COLOR_BGR2RGB)
template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)

scene = scene.astype(np.float64) / 255.0
template = template.astype(np.float64) / 255.0


# Part (b)
# Reduce the scene by a factor of 5

scene_small = cv2.resize(
    scene,
    (
        scene.shape[1] // 5,
        scene.shape[0] // 5
    ),
    interpolation=cv2.INTER_AREA
)

for size in [41, 51, 61]:

    template_resized = resize_template(
        template,
        size
    )

    result = ncc(
        scene_small,
        template_resized
    )

    display_ncc(
        result,
        f"NCC - template size {size}x{size}"
    )


# Part (c)
# Find the region inside the red annulus

mask = find_red_annulus_mask(template)

plt.figure(figsize=(5, 5))
plt.imshow(mask, cmap="gray")
plt.title("Template mask")
plt.axis("off")
plt.show()


# Part (d)

for size in [201, 251, 301]:

    template_resized = resize_template(
        template,
        size
    )

    # Resize the mask along with the template
    mask_resized = cv2.resize(
        mask.astype(np.uint8),
        (size, size),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    result = masked_ncc(
        scene_small,
        template_resized,
        mask_resized
    )

    display_ncc(
        result,
        f"Masked NCC - template size {size}x{size}"
    )