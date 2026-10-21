
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

    # max_candidates=None keeps every valid patch
    if max_candidates is not None and len(positions) > max_candidates:
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
    max_candidates=None, tolerance=1.1, seed=0
):
    """
    Grow a texture outward from the centered input image.
    At each iteration, the unknown pixel whose neighborhood is most
    complete is chosen, and its neighborhood is matched against source
    patches. The selected patch supplies values for unknown pixels in
    that neighborhood.
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

    while not np.all(known):
        # Grow outwards layer by layer: the pixel with the most known
        # neighbours goes next (random among ties). A random frontier
        # order breaks up long structures such as stripes.
        known_count = cv2.boxFilter(
            known.astype(np.float32),
            ddepth=-1,
            ksize=(patch_size, patch_size),
            normalize=False,
            borderType=cv2.BORDER_CONSTANT
        )
        scores = np.where(known, -1, known_count)

        if scores.max() <= 0:
            raise RuntimeError("Texture growth stopped before filling the output.")

        frontier = np.argwhere(scores == scores.max())
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
    image, hole_mask, patch_size=7, search_radius=40,
    tolerance=1.1, seed=1
):
    """
    Fill an arbitrary binary hole using texture patch matching.

    Source patches are all patches lying fully outside the hole within
    search_radius of the neighbourhood being filled, so that each part of
    the hole is filled from the texture that surrounds it (and every
    nearby candidate is compared, not a random subset).
    Pixels outside the hole are preserved exactly.
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

    # valid_source[y, x]: patch with top-left (y, x) lies fully outside
    # the hole (integral image counts the known pixels in every patch)
    integral = cv2.integral((~hole).astype(np.uint8))
    p = patch_size
    known_in_patch = (
        integral[p:, p:] - integral[:-p, p:]
        - integral[p:, :-p] + integral[:-p, :-p]
    )
    valid_source = known_in_patch == p * p

    # windows[y, x] is a view of image[y:y + p, x:x + p] (no copy)
    windows = np.lib.stride_tricks.sliding_window_view(
        image, (p, p), axis=(0, 1)
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

        # Every fully known source patch inside the search window
        # (window grows if it contains none, e.g. deep inside a big hole)
        radius = search_radius
        while True:
            y0, x0 = max(0, py - radius), max(0, px - radius)
            y1 = min(valid_source.shape[0], py + radius + 1)
            x1 = min(valid_source.shape[1], px + radius + 1)
            ys, xs = np.nonzero(valid_source[y0:y1, x0:x1])

            covers_image = (y0, x0) == (0, 0) and (y1, x1) == valid_source.shape
            if len(ys) > 0 or covers_image:
                break
            radius *= 2

        if len(ys) == 0:
            raise ValueError("No fully known source patch is available.")

        candidates = windows[ys + y0, xs + x0].transpose(0, 2, 3, 1)
        candidates = candidates.astype(np.float32)

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


def make_text_mask_auto(
    image, white_min=200, white_spread=30, include_pins=True,
    group_size=15, min_group=150, grow=4, max_island=40
):
    """
    Automatic mask for map labels (white text with a dark outline, and
    purple location pins), kept tight around the label pixels.

    1) Seeds: near-white unsaturated pixels (letters, pin rim and icon)
       and, optionally, saturated purple pixels (pin body).
    2) Seeds are grouped by dilating with a group_size kernel; groups with
       fewer than min_group seed pixels (isolated bright specks on the
       rock) are dropped, so only text-like clusters remain.
    3) The kept seeds are dilated by grow pixels to also cover the dark
       outline and the anti-aliased edges of the letters.
    4) Tiny unmasked islands enclosed by the mask (outline pixels inside
       letters such as 'a', 'e', 'E') are filled.
    """
    rgb = np.asarray(image, dtype=np.int16)
    lowest, highest = rgb.min(axis=2), rgb.max(axis=2)

    seeds = (lowest >= white_min) & (highest - lowest <= white_spread)

    if include_pins:
        hsv = cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2HSV)
        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        seeds |= (hue >= 125) & (hue <= 145) & (sat > 80) & (val > 150)

    groups = cv2.dilate(
        seeds.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (group_size, group_size))
    )
    num_groups, group_id = cv2.connectedComponents(groups)

    seeds_per_group = np.bincount(group_id[seeds], minlength=num_groups)
    keep = seeds_per_group >= min_group
    keep[0] = False

    text = seeds & keep[group_id]

    mask = cv2.dilate(
        text.astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    )

    # Label 0 = masked pixels; every other component is an unmasked region
    _, island_id, stats, _ = cv2.connectedComponentsWithStats(
        (mask == 0).astype(np.uint8), connectivity=4
    )
    small = stats[:, cv2.CC_STAT_AREA] < max_island
    small[0] = False
    mask[small[island_id]] = 255

    return mask


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


def show_pair(left, right, left_title, right_title, save_path, width=12):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Axes widths follow the image widths and the figure is tall enough for
    # both, so the two images share one pixel scale (a small sample is
    # shown small next to a larger image)
    total_width = left.shape[1] + right.shape[1]
    height = width * max(left.shape[0], right.shape[0]) / total_width + 0.6

    fig, axes = plt.subplots(
        1, 2, figsize=(width, height),
        gridspec_kw=dict(width_ratios=[left.shape[1], right.shape[1]])
    )
    axes[0].imshow(left)
    axes[0].set_title(left_title)
    axes[0].axis("off")

    axes[1].imshow(right)
    axes[1].set_title(right_title)
    axes[1].axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir.parent / "data" / "texture"
    # Full-resolution images go to outputs_texture, report figures to output
    output_dir = base_dir / "outputs_texture"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = base_dir.parent / "output"
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Parts (a) and (b), tuned for the three texture samples.
    patch_size = 13
    max_candidates = None  # compare against every patch of the sample
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
            f"Sample\n({sample.shape[0]} x {sample.shape[1]})",
            f"Synthesized from {name} (256 x 256)",
            figure_dir / f"q3b{i + 1}.png"
        )

    # Parts (c) and (d), tuned for lohagad.png.
    inpaint_patch_size = 7
    search_radius = 40
    use_interactive_mask = False  # True: paint the text mask by hand

    lohagad = read_image(data_dir / "lohagad.png")

    if use_interactive_mask:
        text_mask = make_text_mask_interactive(lohagad)
    else:
        text_mask = make_text_mask_auto(lohagad)

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
        patch_size=inpaint_patch_size,
        search_radius=search_radius,
        tolerance=tolerance,
        seed=random_seed + 10
    )

    save_rgb(output_dir / "lohagad_inpainted.png", filled)
    show_pair(
        hole_image,
        filled,
        "Lohagad with text hole",
        "Texture-based inpainting",
        figure_dir / "q3d1.png",
        width=16
    )

if __name__ == "__main__":
    main()
