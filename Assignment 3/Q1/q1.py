import pathlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree, distance


# ---------------------------------------------------------
# (a) Mean-shift segmentation
# ---------------------------------------------------------

def make_features(image, spatial_weight=0.35, smoothing_sigma=0.0):
    """
    Joint feature vector:
    [R, G, B, scaled_X, scaled_Y]

    spatial_weight controls the relative importance of XY.
    smoothing_sigma > 0 Gaussian-smooths the colours first, so that a
    fine texture (e.g. leopard spots on fur) becomes one colour mode.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H, W = image.shape[:2]

    yy, xx = np.mgrid[0:H, 0:W]

    scale = 255.0 / max(H, W)

    colors = image.astype(np.float32)
    if smoothing_sigma > 0:
        colors = cv2.GaussianBlur(colors, (0, 0), smoothing_sigma)

    features = np.dstack([
        colors,
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
    samples = features[indices].astype(np.float64)

    # Gaussian contributions beyond 3h are negligible.
    radius_sq = (3.0 * bandwidth) ** 2

    shifted = samples.copy()

    for _ in range(max_iter):
        # Squared distance from every shifted point to every sample;
        # all points are updated together (vectorized over points)
        squared_dist = distance.cdist(shifted, samples, "sqeuclidean")

        # Gaussian kernel weights
        weights = np.exp(-squared_dist / (2.0 * bandwidth**2))
        weights[squared_dist > radius_sq] = 0.0

        total = np.sum(weights, axis=1, keepdims=True)
        updated = np.where(
            total > 0, weights @ samples / np.maximum(total, 1e-12), shifted
        )

        max_move = np.max(np.linalg.norm(updated - shifted, axis=1))
        shifted = updated

        if max_move < tol:
            break

    return samples, shifted


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


def assign_labels(features, samples, modes, centers):
    """
    Every pixel takes the segment of its nearest density sample, and a
    sample's segment is the merged mode its mean-shift path converged to.
    (Nearest-center assignment would ignore the basins of attraction:
    far-away pixels get pulled to a spatially closer center of the
    wrong colour.)
    Output labels are integers in [1, N].
    """
    sample_labels = np.argmin(distance.cdist(modes, centers), axis=1) + 1

    _, nearest = cKDTree(samples).query(features, workers=-1)

    return sample_labels[nearest].astype(np.int32)


def mean_shift_segmentation(image, spatial_weight=0.35,
                            bandwidth=30, max_iter=15,
                            sample_size=1500, max_segments=10,
                            smoothing_sigma=0.0, tol=0.5):
    """
    Returns:
        labels: H x W integer segmentation map, labels 1..N
        N: number of segments
        rgb: original image in RGB order
    """
    rgb, features = make_features(image, spatial_weight, smoothing_sigma)

    samples, modes = mean_shift_modes(
        features,
        bandwidth=bandwidth,
        max_iter=max_iter,
        tol=tol,
        sample_size=sample_size
    )

    centers = merge_modes(
        modes,
        bandwidth=bandwidth,
        max_segments=max_segments
    )

    labels = assign_labels(features, samples, modes, centers)
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

def display_results(rgb, labels, N, overlay, image_name,
                    output_dir, first_index):
    # Figures are saved as q1c<first_index>.png, q1c<first_index + 1>.png
    # and q1c<first_index + 2>.png in output_dir.
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
    plt.savefig(output_dir / f"q1c{first_index}.png", dpi=150, bbox_inches="tight")
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
            im = ax.imshow(mask, cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"Segment {k}")
        ax.axis("off")

    fig.suptitle(f"{image_name}: Binary segment masks")
    plt.tight_layout()
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    plt.savefig(output_dir / f"q1c{first_index + 1}.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Original image with black boundaries.
    plt.figure(figsize=(8, 6))
    plt.imshow(overlay)
    plt.title(f"{image_name}: Segment boundaries")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / f"q1c{first_index + 2}.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    BASE_DIR = pathlib.Path(__file__).resolve().parent
    DATA_DIR = (
        BASE_DIR.parent
        / "data"
        / "segment"
    )
    OUTPUT_DIR = BASE_DIR.parent / "output"
    OUTPUT_DIR.mkdir(exist_ok=True)

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
        # Spots break the coat into orange/black pixels, so colours are
        # pre-smoothed to make the leopard one colour mode; mean shift is
        # run to convergence (small tol) so grass and leopard modes separate.
        "leopard.png": {
            "spatial_weight": 0.40,
            "bandwidth": 18,
            "max_iter": 300,
            "tol": 0.1,
            "sample_size": 2000,
            "max_segments": 10,
            "smoothing_sigma": 12,
        },
    }

    for i, path in enumerate(image_paths):
        image = cv2.imread(str(path))

        if image is None:
            raise FileNotFoundError(f"Could not load image: {path}")

        name = path.name
        params = PARAMS[name]

        labels, N, rgb = mean_shift_segmentation(image, **params)

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

        display_results(
            rgb, label_image, N, overlay, name, OUTPUT_DIR, 3 * i + 1
        )
