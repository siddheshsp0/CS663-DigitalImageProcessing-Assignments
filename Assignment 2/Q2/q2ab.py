import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from pathlib import Path


def interpolate(image, x, y):


    h, w = image.shape

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1

    if x0 < 0 or x1 >= w or y0 < 0 or y1 >= h:
        return 0.0

    dx = x - x0
    dy = y - y0

    return (
        image[y0, x0] * (1 - dx) * (1 - dy)
        + image[y0, x1] * dx * (1 - dy)
        + image[y1, x0] * (1 - dx) * dy
        + image[y1, x1] * dx * dy
    )


def canny_edge_detection(image, sigma, threshold_low, threshold_high):

    # Gaussian smoothing
    image = image.astype(np.float32)

    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma
    )

    h, w = blurred.shape

    # v(p)
    gx = np.zeros((h, w), dtype=np.float32)
    gy = np.zeros((h, w), dtype=np.float32)

    for y in range(1, h - 1):
        for x in range(1, w - 1):

            gx[y, x] = (
                blurred[y, x + 1]
                - blurred[y, x - 1]
            ) / 2.0

            gy[y, x] = (
                blurred[y + 1, x]
                - blurred[y - 1, x]
            ) / 2.0

    magnitude = np.sqrt(gx * gx + gy * gy)

    nms = np.zeros_like(magnitude)

    # NMS
    for y in range(1, h - 1):
        for x in range(1, w - 1):

            m = magnitude[y, x]

            if m == 0:
                continue

            ux = gx[y, x] / m
            uy = gy[y, x] / m

            m_forward = interpolate(
                magnitude,
                x + ux,
                y + uy
            )

            m_backward = interpolate(
                magnitude,
                x - ux,
                y - uy
            )

            if m > m_forward and m > m_backward:
                nms[y, x] = m

    # print(f"Thresholds: low={threshold_low}, high={threshold_high}", nms.max(), nms.min(), nms.mean())
    strong = nms > threshold_high
    weak = (
        (nms > threshold_low)
        & (nms <= threshold_high)
    )

    edges = np.zeros((h, w), dtype=np.uint8)

    edges[strong] = 255
    queue = deque()

    ys, xs = np.where(strong)

    for y, x in zip(ys, xs):
        queue.append((y, x))

    while queue:

        y, x = queue.popleft()

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):

                if dx == 0 and dy == 0:
                    continue

                ny = y + dy
                nx = x + dx

                if (
                    ny < 0 or ny >= h
                    or nx < 0 or nx >= w
                ):
                    continue

                if weak[ny, nx]:

                    weak[ny, nx] = False
                    edges[ny, nx] = 255

                    queue.append((ny, nx))

    return edges


def draw_edges_on_image(image, edges):

    output = image.copy()

    # Detected edge pixels become black.
    output[edges > 0] = [0, 0, 0]

    return output


def process_image(path, sigma, threshold_low, threshold_high):

    # Grayscale image for Canny
    gray = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if gray is None:
        raise FileNotFoundError(path)

    # Original colour image for final visualization
    original = cv2.imread(str(path))

    # OpenCV stores colour images as BGR.
    original_rgb = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2RGB
    )

    # Canny
    edges = canny_edge_detection(
        gray,
        sigma,
        threshold_low,
        threshold_high
    )

    # Edges drawn in black on original image
    overlay = draw_edges_on_image(
        original,
        edges
    )

    overlay_rgb = cv2.cvtColor(
        overlay,
        cv2.COLOR_BGR2RGB
    )

    return original_rgb, edges, overlay_rgb


# Main program
params = {
    "butterfly.png":    (1.0, 20, 35),
    "paithaniEdge.png": (1.3, 10, 25),
    "rangoli.png":      (1.2, 5, 12)
}


data_dir = (
    Path(__file__).resolve().parent.parent / "data" / "edge" )

images = [
    data_dir / "butterfly.png",
    data_dir / "paithaniEdge.png",
    data_dir / "rangoli.png"
]


for path in images:

    name = path.name

    sigma, low, high = params[name]

    original, edges, overlay = process_image(
        path,
        sigma,
        low,
        high
    )
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(edges, cmap="gray", vmin=0, vmax=255)
    plt.title("Binary Canny Edges")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Edges on Input")
    plt.axis("off")

    plt.suptitle(
        f"{name}    sigma={sigma}, "
        f"Low={low}, High={high}"
    )

    plt.tight_layout()
    plt.show()