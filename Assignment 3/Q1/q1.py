import pathlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree, distance


# ---------------------------------------------------------
# (a) Mean-shift segmentation
# ---------------------------------------------------------

def make_features(image, spatial_weight=0.35):
    """
    Joint feature vector:
    [R, G, B, scaled_X, scaled_Y]

    spatial_weight controls the relative importance of XY.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H, W = image.shape[:2]

    yy, xx = np.mgrid[0:H, 0:W]

    scale = 255.0 / max(H, W)

    features = np.dstack([
        image.astype(np.float32),
        xx.astype(np.float32) * scale * spatial_weight,
        yy.astype(np.float32) * scale * spatial_weight
    ])

    return image, features.reshape(-1, 5)


def mean_shift_modes(features, bandwidth=35, max_iter=15,
                     tol=0.5, sample_size=1500, seed=0):
    """
    Gaussian-kernel mean shift using a subset of pixels to
    approximate the feature density.

    The density samples remain fixed during iterations,
    following the fixed-PDF option in the slides.
    """
    rng = np.random.default_rng(seed)

    n = len(features)
    count = min(sample_size, n)

    # Approximately uniform sampling over the feature observations
    indices = rng.choice(n, size=count, replace=False)
    samples = features[indices].copy()

    # KD-tree accelerates neighborhood searches.
    # Gaussian contributions beyond 3h are negligible.
    tree = cKDTree(samples)
    radius = 3.0 * bandwidth

    shifted = samples.copy()

    for _ in range(max_iter):
        max_move = 0.0
        updated = shifted.copy()

        for i, point in enumerate(shifted):
            neighbors = tree.query_ball_point(point, radius)

            if not neighbors:
                continue

            neighbors = np.asarray(neighbors, dtype=np.int32)
            points = samples[neighbors]

            diff = points - point
            squared_dist = np.sum(diff * diff, axis=1)

            # Gaussian kernel weights
            weights = np.exp(
                -squared_dist / (2.0 * bandwidth**2)
            )

            updated[i] = np.sum(
                points * weights[:, None], axis=0
            ) / (np.sum(weights) + 1e-12)

            movement = np.linalg.norm(updated[i] - point)
            max_move = max(max_move, movement)

        shifted = updated

        if max_move < tol:
            break

    return shifted


def merge_modes(modes, bandwidth=35, max_segments=10):
    """
    Merge converged modes that are close in joint feature space.
    If more than max_segments remain, merge the closest pair
    until the requested upper bound is reached.
    """
    # First merge modes within a small tolerance.
    clustering_tree = cKDTree(modes)
    pairs = clustering_tree.query_pairs(r=0.6 * bandwidth)

    parent = np.arange(len(modes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    for a, b in pairs:
        union(a, b)

    groups = {}
    for i in range(len(modes)):
        root = find(i)
        groups.setdefault(root, []).append(i)

    centers = np.array([
        modes[group].mean(axis=0)
        for group in groups.values()
    ])

    # Enforce the assignment's desired upper bound.
    while len(centers) > max_segments:
        D = distance.cdist(centers, centers)
        np.fill_diagonal(D, np.inf)

        i, j = np.unravel_index(np.argmin(D), D.shape)

        merged = (centers[i] + centers[j]) / 2.0
        keep = np.ones(len(centers), dtype=bool)
        keep[[i, j]] = False

        centers = np.vstack([centers[keep], merged])

    return centers


def assign_labels(features, centers, chunk_size=50000):
    """
    Assign every image pixel to its nearest converged mode.
    Output labels are integers in [1, N].
    """
    labels = np.empty(len(features), dtype=np.int32)

    for start in range(0, len(features), chunk_size):
        end = min(start + chunk_size, len(features))

        D = distance.cdist(features[start:end], centers)
        labels[start:end] = np.argmin(D, axis=1) + 1

    return labels


def mean_shift_segmentation(image, spatial_weight=0.35,
                            bandwidth=30, max_iter=15,
                            sample_size=1500, max_segments=10):
    """
    Returns:
        labels: H x W integer segmentation map, labels 1..N
        N: number of segments
        rgb: original image in RGB order
    """
    rgb, features = make_features(image, spatial_weight)

    modes = mean_shift_modes(
        features,
        bandwidth=bandwidth,
        max_iter=max_iter,
        sample_size=sample_size
    )

    centers = merge_modes(
        modes,
        bandwidth=bandwidth,
        max_segments=max_segments
    )

    labels = assign_labels(features, centers)
    labels = labels.reshape(image.shape[:2])

    return labels, len(centers), rgb


# ---------------------------------------------------------
# (b) Label image and segment boundaries
# ---------------------------------------------------------

def create_label_image(image, labels):
    """
    Returns an H x W label image whose pixel values are 1..N.
    """
    return labels.astype(np.int32)


def draw_segment_boundaries(image, labels, thickness=2):
    """
    Draw black boundaries between neighboring segments.
    """
    boundary = np.zeros(labels.shape, dtype=np.uint8)

    # Detect horizontal and vertical label changes.
    boundary[:, 1:] |= (
        labels[:, 1:] != labels[:, :-1]
    ).astype(np.uint8)

    boundary[1:, :] |= (
        labels[1:, :] != labels[:-1, :]
    ).astype(np.uint8)

    # Dilate to make boundaries thicker.
    # A 2x2 kernel gives approximately 2-pixel thickness.
    kernel = np.ones((2, 2), dtype=np.uint8)
    boundary = cv2.dilate(
        boundary, kernel, iterations=max(1, thickness // 2)
    )

    # Work in RGB for display.
    result = image.copy()
    result[boundary > 0] = [0, 0, 0]

    return result


# ---------------------------------------------------------
# (c) Display original, segmentation masks, and boundaries
# ---------------------------------------------------------

def display_results(rgb, labels, N, overlay, image_name):
    # Color-coded segmentation map for convenient viewing.
    label_display = np.zeros_like(rgb)

    rng = np.random.default_rng(10)
    colors = rng.integers(30, 256, size=(N, 3), dtype=np.uint8)

    for k in range(1, N + 1):
        label_display[labels == k] = colors[k - 1]

    # Original image alongside the segmentation map.
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(rgb)
    plt.title(f"{image_name}: Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(label_display)
    plt.title(f"Segmentation map ({N} segments)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # Binary masks: one black-and-white image per segment.
    cols = min(5, N)
    rows = (N + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False
    )

    for k, ax in enumerate(axes.flat, start=1):
        if k <= N:
            mask = (labels == k).astype(np.uint8) * 255
            ax.imshow(mask, cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"Segment {k}")
        ax.axis("off")

    fig.suptitle(f"{image_name}: Binary segment masks")
    plt.tight_layout()
    plt.show()

    # Original image with black boundaries.
    plt.figure(figsize=(8, 6))
    plt.imshow(overlay)
    plt.title(f"{image_name}: Segment boundaries")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    BASE_DIR = pathlib.Path(__file__).resolve().parent
    DATA_DIR = (
        BASE_DIR.parent
        / "data"
        / "segment"
    )

    image_paths = [
        DATA_DIR / "bison.png",
        DATA_DIR / "ghats.png",
        DATA_DIR / "leopard.png",
    ]

    # Different parameters for each image
    PARAMS = {
        "bison.png": {
            "spatial_weight": 0.20,
            "bandwidth": 30,
            "max_iter": 20,
            "sample_size": 1500,
            "max_segments": 15,
        },
        "ghats.png": {
            "spatial_weight": 0.50,
            "bandwidth": 30,
            "max_iter": 15,
            "sample_size": 2000,
            "max_segments": 10,
        },
        "leopard.png": {
            "spatial_weight": 0.35,
            "bandwidth": 35,
            "max_iter": 15,
            "sample_size": 1500,
            "max_segments": 10,
        },
    }

    for path in image_paths:
        image = cv2.imread(str(path))

        if image is None:
            raise FileNotFoundError(f"Could not load image: {path}")

        name = path.name
        params = PARAMS[name]

        labels, N, rgb = mean_shift_segmentation(
            image,
            spatial_weight=params["spatial_weight"],
            bandwidth=params["bandwidth"],
            max_iter=params["max_iter"],
            sample_size=params["sample_size"],
            max_segments=params["max_segments"],
        )

        label_image = create_label_image(image, labels)
        overlay = draw_segment_boundaries(
            rgb, label_image, thickness=2
        )

        print(f"{name}: {N} segments")
        print(
            "Label range:",
            int(label_image.min()),
            "to",
            int(label_image.max())
        )

        display_results(rgb, label_image, N, overlay, name)
