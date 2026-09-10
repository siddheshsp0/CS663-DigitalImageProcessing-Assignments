import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# =============================================================
# Structure Tensor
# =============================================================

def compute_structure_tensor(image, sigma_gradient, sigma_tensor):

    image = image.astype(np.float32)

    # Smooth image before computing gradients
    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=sigma_gradient,
        sigmaY=sigma_gradient
    )

    h, w = blurred.shape

    # Compute Ix and Iy using central differences
    ix = np.zeros((h, w), dtype=np.float32)
    iy = np.zeros((h, w), dtype=np.float32)

    for y in range(1, h - 1):
        for x in range(1, w - 1):

            ix[y, x] = (
                blurred[y, x + 1]
                - blurred[y, x - 1]
            ) / 2.0

            iy[y, x] = (
                blurred[y + 1, x]
                - blurred[y - 1, x]
            ) / 2.0

    # Components of the structure tensor
    #
    # A = [ sum(Ix^2)    sum(IxIy) ]
    #     [ sum(IxIy)    sum(Iy^2) ]
    #
    # Gaussian weighting is implemented using GaussianBlur.
    a11 = cv2.GaussianBlur(
        ix * ix,
        (0, 0),
        sigmaX=sigma_tensor,
        sigmaY=sigma_tensor
    )

    a12 = cv2.GaussianBlur(
        ix * iy,
        (0, 0),
        sigmaX=sigma_tensor,
        sigmaY=sigma_tensor
    )

    a22 = cv2.GaussianBlur(
        iy * iy,
        (0, 0),
        sigmaX=sigma_tensor,
        sigmaY=sigma_tensor
    )

    # Eigenvalues and eigenvectors at every pixel
    #
    # np.linalg.eigh() is used because A is symmetric.
    eigenvalues = np.zeros((h, w, 2), dtype=np.float32)
    eigenvectors = np.zeros((h, w, 2, 2), dtype=np.float32)

    for y in range(h):
        for x in range(w):

            A = np.array([
                [a11[y, x], a12[y, x]],
                [a12[y, x], a22[y, x]]
            ], dtype=np.float32)

            values, vectors = np.linalg.eigh(A)

            # np.linalg.eigh gives ascending order.
            # Store largest eigenvalue first.
            eigenvalues[y, x, 0] = values[1]
            eigenvalues[y, x, 1] = values[0]

            eigenvectors[y, x, :, 0] = vectors[:, 1]
            eigenvectors[y, x, :, 1] = vectors[:, 0]

    return (
        ix,
        iy,
        a11,
        a12,
        a22,
        eigenvalues,
        eigenvectors
    )


# =============================================================
# Harris-Stephens cornerness
# =============================================================

def harris_cornerness(a11, a12, a22, k):

    determinant = a11 * a22 - a12 * a12
    trace = a11 + a22

    C = determinant - k * trace * trace

    return C


# =============================================================
# Non-maximum suppression
# =============================================================

def non_maximum_suppression(image, size=3):

    h, w = image.shape

    result = np.zeros_like(image)

    radius = size // 2

    for y in range(radius, h - radius):
        for x in range(radius, w - radius):

            value = image[y, x]

            if value <= 0:
                continue

            neighborhood = image[
                y - radius:y + radius + 1,
                x - radius:x + radius + 1
            ]

            if value == np.max(neighborhood):
                result[y, x] = value

    return result


# =============================================================
# Harris-Stephens edge-ness
#
# Harris C < 0 corresponds to an edge.
#
# Convert it to positive edge strength:
#
#       edge_strength = -C
#
# so that larger values represent stronger edges.
# =============================================================

def harris_edge_strength(C):

    edge_strength = np.zeros_like(C)

    edge_strength[C < 0] = -C[C < 0]

    return edge_strength


# =============================================================
# Threshold
# =============================================================

def threshold_response(response, threshold):

    output = np.zeros_like(response, dtype=np.uint8)

    output[response > threshold] = 255

    return output


# =============================================================
# Draw detected pixels on image
# =============================================================

def draw_corners_on_image(image, corners):

    output = image.copy()

    output[corners > 0] = [0, 0, 0]

    return output


def draw_edges_on_image(image, edges):

    output = image.copy()

    output[edges > 0] = [255, 255, 255]

    return output


# =============================================================
# Process one image
# =============================================================

def process_image(
    path,
    sigma_gradient,
    sigma_tensor,
    harris_k,
    harris_corner_threshold,
    shi_tomasi_threshold,
    harris_edge_threshold
):

    # Read image
    gray = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    original = cv2.imread(str(path))

    if gray is None or original is None:
        raise FileNotFoundError(path)

    original_rgb = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2RGB
    )

    # 1. Structure tensor
    (
        ix,
        iy,
        a11,
        a12,
        a22,
        eigenvalues,
        eigenvectors
    ) = compute_structure_tensor(
        gray,
        sigma_gradient,
        sigma_tensor
    )

    # Eigenvalues
    lambda1 = eigenvalues[:, :, 0]
    lambda2 = eigenvalues[:, :, 1]

    # 2. Harris-Stephens cornerness
    C = harris_cornerness(
        a11,
        a12,
        a22,
        harris_k
    )

    # 3. Harris corner NMS
    harris_corner_response = np.zeros_like(C)

    harris_corner_response[C > 0] = C[C > 0]

    harris_corner_nms = non_maximum_suppression(
        harris_corner_response
    )

    # 4. Shi-Tomasi
    shi_tomasi = lambda2.copy()

    shi_tomasi_nms = non_maximum_suppression(
        shi_tomasi
    )

    # 5. Binary corner outputs
    harris_corners = threshold_response(
        harris_corner_nms,
        harris_corner_threshold
    )

    shi_tomasi_corners = threshold_response(
        shi_tomasi_nms,
        shi_tomasi_threshold
    )

    # 6. Harris edge-ness
    # C < 0 -> edge
    harris_edges_strength = harris_edge_strength(C)

    harris_edges_nms = non_maximum_suppression(
        harris_edges_strength
    )

    # 7. Binary Harris edge output
    harris_edges = threshold_response(
        harris_edges_nms,
        harris_edge_threshold
    )

    # 8. Draw corners in black
    harris_corner_overlay = draw_corners_on_image(
        original,
        harris_corners
    )

    shi_tomasi_corner_overlay = draw_corners_on_image(
        original,
        shi_tomasi_corners
    )

    # 9. Draw Harris edges in white
    harris_edge_overlay = draw_edges_on_image(
        original,
        harris_edges
    )

    # Convert overlays for matplotlib
    harris_corner_overlay = cv2.cvtColor(
        harris_corner_overlay,
        cv2.COLOR_BGR2RGB
    )

    shi_tomasi_corner_overlay = cv2.cvtColor(
        shi_tomasi_corner_overlay,
        cv2.COLOR_BGR2RGB
    )

    harris_edge_overlay = cv2.cvtColor(
        harris_edge_overlay,
        cv2.COLOR_BGR2RGB
    )

    return {
        "original": original_rgb,
        "lambda1": lambda1,
        "lambda2": lambda2,
        "C": C,
        "harris_corner_nms": harris_corner_nms,
        "shi_tomasi_nms": shi_tomasi_nms,
        "harris_corners": harris_corners,
        "shi_tomasi_corners": shi_tomasi_corners,
        "harris_corner_overlay": harris_corner_overlay,
        "shi_tomasi_corner_overlay": shi_tomasi_corner_overlay,
        "harris_edge_nms": harris_edges_nms,
        "harris_edges": harris_edges,
        "harris_edge_overlay": harris_edge_overlay
    }


# Parameters
params = {

    "nandadevi.png": {
        "sigma_gradient": 1.0,
        "sigma_tensor": 2.0,
        "harris_k": 0.04,
        "harris_corner_threshold": 1000,
        "shi_tomasi_threshold": 100,
        "harris_edge_threshold": 100
    },

    "paithaniCorner.png": {
        "sigma_gradient": 1.0,
        "sigma_tensor": 2.0,
        "harris_k": 0.04,
        "harris_corner_threshold": 1000,
        "shi_tomasi_threshold": 100,
        "harris_edge_threshold": 100
    },

    "warli.png": {
        "sigma_gradient": 1.0,
        "sigma_tensor": 2.0,
        "harris_k": 0.04,
        "harris_corner_threshold": 1000,
        "shi_tomasi_threshold": 100,
        "harris_edge_threshold": 100
    }
}


# Input images
data_dir = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "corner"
)

images = [
    data_dir / "nandadevi.png",
    data_dir / "paithaniCorner.png",
    data_dir / "warli.png"
]


# Display results
for path in images:

    name = path.name
    p = params[name]

    results = process_image(
        path,
        p["sigma_gradient"],
        p["sigma_tensor"],
        p["harris_k"],
        p["harris_corner_threshold"],
        p["shi_tomasi_threshold"],
        p["harris_edge_threshold"]
    )

    # Normalize continuous-valued images only for display
    lambda1_display = cv2.normalize(
        results["lambda1"],
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    lambda2_display = cv2.normalize(
        results["lambda2"],
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    C_display = cv2.normalize(
        results["C"],
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    harris_nms_display = cv2.normalize(
        results["harris_corner_nms"],
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    shi_nms_display = cv2.normalize(
        results["shi_tomasi_nms"],
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    edge_nms_display = cv2.normalize(
        results["harris_edge_nms"],
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    # Figure 1: Required outputs (1) - (5)
    plt.figure(figsize=(15, 12))

    plt.subplot(2, 3, 1)
    plt.imshow(results["original"])
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(lambda1_display, cmap="gray")
    plt.title("Largest Eigenvalue")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(lambda2_display, cmap="gray")
    plt.title("Second-largest Eigenvalue\nShi-Tomasi Cornerness")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(C_display, cmap="gray")
    plt.title("Harris-Stephens Cornerness C")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(harris_nms_display, cmap="gray")
    plt.title("Harris Cornerness after NMS")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.imshow(shi_nms_display, cmap="gray")
    plt.title("Shi-Tomasi Cornerness after NMS")
    plt.axis("off")

    plt.suptitle(name)
    plt.tight_layout()
    plt.show()

    # Figure 2: Required outputs (6) - (7)
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(results["harris_corners"], cmap="gray")
    plt.title("Binary Harris-Stephens Corners")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(results["shi_tomasi_corners"], cmap="gray")
    plt.title("Binary Shi-Tomasi Corners")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(results["harris_corner_overlay"])
    plt.title("Harris Corners on Input")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(results["shi_tomasi_corner_overlay"])
    plt.title("Shi-Tomasi Corners on Input")
    plt.axis("off")

    plt.suptitle(name)
    plt.tight_layout()
    plt.show()

    # Figure 3: Required outputs (8) - (10)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(edge_nms_display, cmap="gray")
    plt.title("Harris Edge-ness after NMS")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(results["harris_edges"], cmap="gray")
    plt.title("Binary Harris-Stephens Edges")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(results["harris_edge_overlay"])
    plt.title("Harris Edges on Input")
    plt.axis("off")

    plt.suptitle(name)
    plt.tight_layout()
    plt.show()