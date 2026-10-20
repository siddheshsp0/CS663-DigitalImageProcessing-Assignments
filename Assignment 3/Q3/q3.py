
from pathlib import Path
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt

TEXTURE_NAMES = [
    "textureSample81.png",
    "textureSample110.png",
    "textureSample207.png",
]

def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_rgb(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.clip(image, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def make_weights(patch_size):
    """Gaussian spatial weights, centered on the patch."""
    coords = np.arange(patch_size) - patch_size // 2
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    sigma = max(1.0, patch_size / 3.0)
    weights = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return weights.astype(np.float32)


def extract_source_patches(
    texture, patch_size, max_candidates=200, rng=None, valid_mask=None
):
    """
    Extract a random subset of valid source patches.
    Integral images are used to find valid patch locations without
    checking every patch with a Python loop.
    """
    if rng is None:
        rng = np.random.default_rng()

    texture = np.asarray(texture, dtype=np.uint8)
    h, w = texture.shape[:2]

    if valid_mask is None:
        valid_mask = np.ones((h, w), dtype=np.uint8)
    else:
        valid_mask = (np.asarray(valid_mask) > 0).astype(np.uint8)

    # Pad small source images and their masks consistently.
    if h < patch_size or w < patch_size:
        pad_y = max(0, patch_size - h)
        pad_x = max(0, patch_size - w)
        top, bottom = pad_y // 2, pad_y - pad_y // 2
        left, right = pad_x // 2, pad_x - pad_x // 2

        texture = cv2.copyMakeBorder(
            texture, top, bottom, left, right, cv2.BORDER_REFLECT
        )
        valid_mask = cv2.copyMakeBorder(
            valid_mask, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=0
        )
        h, w = texture.shape[:2]

    # Integral image gives the number of valid pixels in each patch.
    integral = cv2.integral(valid_mask)
    sums = (
        integral[patch_size:, patch_size:]
        - integral[:-patch_size, patch_size:]
        - integral[patch_size:, :-patch_size]
        + integral[:-patch_size, :-patch_size]
    )

    ys, xs = np.where(sums == patch_size * patch_size)

    positions = np.column_stack((ys, xs))

    if len(positions) > max_candidates:
        selected = rng.choice(
            len(positions), size=max_candidates, replace=False
        )
        positions = positions[selected]

    patches = np.stack([
        texture[y:y + patch_size, x:x + patch_size]
        for y, x in positions
    ]).astype(np.float32)

    return patches, positions, texture


def choose_patch(
    candidates, target_patch, known_mask, spatial_weights,
    rng, tolerance=1.1
):
    """
    Find source patches that best match the known target neighborhood.
    Select randomly among candidates with distance <= tolerance * best.
    """
    valid = known_mask
    if not np.any(valid):
        return int(rng.integers(len(candidates)))

    # Shape: candidates x known-pixels x RGB channels.
    differences = (
        candidates[:, valid, :]
        - target_patch[valid, :][None, :, :]
    )

    squared_rgb_error = np.sum(differences * differences, axis=2)
    weights = spatial_weights[valid]

    distances = np.sum(
        squared_rgb_error * weights[None, :], axis=1
    ) / (3.0 * np.sum(weights))

    best = float(np.min(distances))

    if not np.isfinite(best):
        return int(rng.integers(len(candidates)))

    if best < 1e-8:
        eligible = np.flatnonzero(distances <= best + 1e-8)
    else:
        eligible = np.flatnonzero(distances <= tolerance * best)

    if len(eligible) == 0:
        return int(np.argmin(distances))

    return int(rng.choice(eligible))


def patch_bounds(y, x, patch_size, height, width):
    """Return valid top-left patch coordinates, including at borders."""
    if height < patch_size or width < patch_size:
        raise ValueError(
            "The output image must be at least as large as patch_size."
        )

    y0 = int(np.clip(y - patch_size // 2, 0, height - patch_size))
    x0 = int(np.clip(x - patch_size // 2, 0, width - patch_size))
    return y0, x0


def grow_texture(
    texture, output_size=256, patch_size=5,
    max_candidates=200, tolerance=1.1, seed=0
):
    """
    Grow a texture outward from the centered input image.
    At each iteration, a frontier neighborhood is matched against
    source patches. The selected patch supplies values for unknown
    pixels in that neighborhood.
    """
    rng = np.random.default_rng(seed)
    texture = np.asarray(texture, dtype=np.uint8)

    if patch_size < 1 or patch_size % 2 == 0:
        raise ValueError("patch_size must be a positive odd integer.")

    height = width = int(output_size)
    if height < patch_size or width < patch_size:
        raise ValueError("output_size must be at least patch_size.")

    candidates, _, source = extract_source_patches(
        texture, patch_size, max_candidates, rng
    )
    weights = make_weights(patch_size)

    # Center the entire provided sample image in the output.
    sh, sw = source.shape[:2]
    if sh > height or sw > width:
        scale = min(height / sh, width / sw)
        new_w = max(1, int(sw * scale))
        new_h = max(1, int(sh * scale))
        source = cv2.resize(
            source, (new_w, new_h), interpolation=cv2.INTER_AREA
        )
        sh, sw = source.shape[:2]

    output = np.zeros((height, width, 3), dtype=np.uint8)
    known = np.zeros((height, width), dtype=bool)

    y0 = (height - sh) // 2
    x0 = (width - sw) // 2
    output[y0:y0 + sh, x0:x0 + sw] = source
    known[y0:y0 + sh, x0:x0 + sw] = True

    kernel = np.ones((3, 3), dtype=np.uint8)

    while not np.all(known):
        unknown = ~known
        adjacent = cv2.dilate(known.astype(np.uint8), kernel) > 0
        frontier = np.argwhere(unknown & adjacent)

        if len(frontier) == 0:
            raise RuntimeError("Texture growth stopped before filling the output.")

        # Random frontier order avoids always growing in one direction.
        cy, cx = frontier[rng.integers(len(frontier))]
        py, px = patch_bounds(
            cy, cx, patch_size, height, width
        )

        target = output[py:py + patch_size, px:px + patch_size].astype(
            np.float32
        )
        local_known = known[py:py + patch_size, px:px + patch_size]

        index = choose_patch(
            candidates, target, local_known, weights, rng, tolerance
        )
        source_patch = candidates[index].astype(np.uint8)

        region = output[py:py + patch_size, px:px + patch_size]
        region[~local_known] = source_patch[~local_known]
        known[py:py + patch_size, px:px + patch_size] = True

    return output


def inpaint_texture(
    image, hole_mask, texture=None, patch_size=5,
    max_candidates=200, tolerance=1.1, seed=1
):
    """
    Fill an arbitrary binary hole using texture patch matching.

    If texture is None, patches come from fully known regions of the
    input image. Pixels outside the hole are preserved exactly.
    """
    rng = np.random.default_rng(seed)
    image = np.asarray(image, dtype=np.uint8)
    hole = np.asarray(hole_mask) > 0

    if hole.shape != image.shape[:2]:
        raise ValueError("hole_mask must match the image dimensions.")
    if patch_size < 1 or patch_size % 2 == 0:
        raise ValueError("patch_size must be a positive odd integer.")
    if not np.any(hole):
        return image.copy()

    height, width = hole.shape
    if height < patch_size or width < patch_size:
        raise ValueError("Image dimensions must be at least patch_size.")

    source = image if texture is None else np.asarray(texture, dtype=np.uint8)
    if texture is None:
        source_valid = ~hole
    else:
        source_valid = np.ones(source.shape[:2], dtype=bool)

    candidates, _, _ = extract_source_patches(
        source, patch_size, max_candidates, rng, source_valid
    )
    weights = make_weights(patch_size)

    output = image.copy()
    known = ~hole
    kernel = np.ones((3, 3), dtype=np.uint8)

    # Rank frontier pixels using a box filter rather than a Python loop.
    while np.any(~known):
        unknown = ~known
        adjacent = cv2.dilate(known.astype(np.uint8), kernel) > 0
        frontier_mask = unknown & adjacent

        if not np.any(frontier_mask):
            raise ValueError(
                "The hole has no known boundary pixels to guide inpainting."
            )

        neighbor_counts = cv2.boxFilter(
            known.astype(np.float32),
            ddepth=-1,
            ksize=(patch_size, patch_size),
            normalize=False,
            borderType=cv2.BORDER_CONSTANT
        )
        scores = np.where(frontier_mask, neighbor_counts, -1)
        best_score = np.max(scores)
        frontier = np.argwhere(scores == best_score)
        cy, cx = frontier[rng.integers(len(frontier))]

        py, px = patch_bounds(
            cy, cx, patch_size, height, width
        )
        target = output[py:py + patch_size, px:px + patch_size].astype(
            np.float32
        )
        local_known = known[py:py + patch_size, px:px + patch_size]

        index = choose_patch(
            candidates, target, local_known, weights, rng, tolerance
        )
        source_patch = candidates[index].astype(np.uint8)

        region = output[py:py + patch_size, px:px + patch_size]
        local_hole = ~local_known
        region[local_hole] = source_patch[local_hole]
        known[py:py + patch_size, px:px + patch_size][local_hole] = True

    # Strictly preserve every pixel outside the requested mask.
    output[~hole] = image[~hole]
    return output


def make_text_mask_interactive(image):
    """Paint a tight mask over the text. Enter accepts; u undoes; r resets."""
    base = image.copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    brush = [5]
    history = []
    drawing = [False]
    last_point = [None]
    window = "Paint over text | ENTER=accept, u=undo, r=reset, [ ]=brush"

    def redraw():
        overlay = base.copy()
        selected = mask > 0
        overlay[selected] = (
            0.35 * overlay[selected]
            + 0.65 * np.array([255, 40, 40])
        ).astype(np.uint8)
        cv2.imshow(window, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    def mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            history.append(mask.copy())
            drawing[0] = True
            last_point[0] = (x, y)
            cv2.circle(mask, (x, y), brush[0], 255, -1)
            redraw()

        elif event == cv2.EVENT_MOUSEMOVE and drawing[0]:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                cv2.line(
                    mask, last_point[0], (x, y), 255,
                    thickness=2 * brush[0]
                )
                cv2.circle(mask, (x, y), brush[0], 255, -1)
                last_point[0] = (x, y)
                redraw()

        elif event == cv2.EVENT_LBUTTONUP:
            drawing[0] = False
            last_point[0] = None

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, mouse)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 10):
            break
        elif key == ord("u") and history:
            mask[:] = history.pop()
            redraw()
        elif key == ord("r"):
            history.append(mask.copy())
            mask[:] = 0
            redraw()
        elif key == ord("]"):
            brush[0] = min(40, brush[0] + 2)
        elif key == ord("["):
            brush[0] = max(1, brush[0] - 2)
        elif key == 27:
            cv2.destroyAllWindows()
            raise KeyboardInterrupt("Masking cancelled.")

    cv2.destroyAllWindows()

    # Small dilation removes residual anti-aliased text edges.
    mask = cv2.dilate(
        mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    return mask


def show_pair(left, right, left_title, right_title, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(left)
    axes[0].set_title(left_title)
    axes[0].axis("off")

    axes[1].imshow(right)
    axes[1].set_title(right_title)
    axes[1].axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir.parent / "data" / "texture"
    output_dir = base_dir / "outputs_texture"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tune these parameters after inspecting each output.
    patch_size = 5
    max_candidates = 200
    tolerance = 1.1
    random_seed = 0

    # Parts (a) and (b).
    for i, name in enumerate(TEXTURE_NAMES):
        sample = read_image(data_dir / name)

        result = grow_texture(
            sample,
            output_size=256,
            patch_size=patch_size,
            max_candidates=max_candidates,
            tolerance=tolerance,
            seed=random_seed + i
        )

        save_rgb(output_dir / f"synth_{Path(name).stem}.png", result)
        show_pair(
            sample,
            result,
            f"Input: {name}",
            "Synthesized 256 x 256",
            output_dir / f"comparison_{Path(name).stem}.png"
        )

    # Parts (c) and (d).
    lohagad = read_image(data_dir / "lohagad.png")
    text_mask = make_text_mask_interactive(lohagad)

    save_rgb(
        output_dir / "lohagad_hole_mask.png",
        np.repeat(text_mask[:, :, None], 3, axis=2)
    )

    hole_image = lohagad.copy()
    hole_image[text_mask > 0] = 0
    save_rgb(output_dir / "lohagad_with_hole.png", hole_image)

    filled = inpaint_texture(
        lohagad,
        text_mask,
        texture=None,
        patch_size=patch_size,
        max_candidates=max_candidates,
        tolerance=tolerance,
        seed=random_seed + 10
    )

    save_rgb(output_dir / "lohagad_inpainted.png", filled)
    show_pair(
        hole_image,
        filled,
        "Lohagad with text hole",
        "Texture-based inpainting",
        output_dir / "lohagad_inpainting_comparison.png"
    )

if __name__ == "__main__":
    main()
