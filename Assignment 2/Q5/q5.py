import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path


# ============================================================
# 1. Manual foreground selection
# ============================================================

def select_foreground(image, title):
    """
    Display image and allow the user to click polygon vertices
    around the foreground object.

    Left click  : add point
    Right click : finish polygon

    Returns:
        foreground_mask : bool array, True for foreground
    """

    fig, ax = plt.subplots(figsize=(10, 8))

    if image.ndim == 2:
        ax.imshow(image, cmap="gray")
    else:
        ax.imshow(image)

    ax.set_title(
        title +
        "\nLeft click: polygon points | Right click: finish"
    )

    points = []

    def onclick(event):
        if event.inaxes != ax:
            return

        if event.button == 1:
            # Left click: add point
            points.append((event.xdata, event.ydata))

            ax.plot(
                event.xdata,
                event.ydata,
                "ro",
                markersize=4
            )

            if len(points) > 1:
                x = [p[0] for p in points[-2:]]
                y = [p[1] for p in points[-2:]]
                ax.plot(x, y, "r-")

            fig.canvas.draw_idle()

        elif event.button == 3:
            # Right click: finish
            if len(points) < 3:
                print("Need at least 3 points.")
                return

            # Close polygon visually
            x = [p[0] for p in points] + [points[0][0]]
            y = [p[1] for p in points] + [points[0][1]]
            ax.plot(x, y, "r-")

            fig.canvas.draw_idle()
            plt.disconnect(cid)
            plt.close(fig)

    cid = fig.canvas.mpl_connect("button_press_event", onclick)

    plt.show()

    if len(points) < 3:
        raise ValueError("No valid polygon was selected.")

    # --------------------------------------------------------
    # Convert polygon to a binary mask.
    # --------------------------------------------------------

    h, w = image.shape[:2]

    yy, xx = np.mgrid[0:h, 0:w]

    coordinates = np.column_stack(
        (xx.ravel(), yy.ravel())
    )

    polygon = Path(points)

    foreground_mask = polygon.contains_points(
        coordinates
    ).reshape(h, w)

    return foreground_mask


# ============================================================
# 2. Exact Euclidean distance transform
#    (implemented ourselves)
# ============================================================

def distance_transform(binary):
    """
    Compute Euclidean distance to the nearest True pixel.

    Implements the 1D squared-distance transform using the
    lower-envelope/parabola algorithm and applies it twice.

    No scipy/skimage/OpenCV is used.
    """

    h, w = binary.shape

    INF = 10**12

    # f = 0 at target pixels, infinity elsewhere
    f = np.where(binary, 0.0, float(INF))

    # --------------------------------------------------------
    # 1D squared distance transform
    # --------------------------------------------------------

    def dt_1d(f):
        n = len(f)

        d = np.empty(n, dtype=np.float64)

        # Locations of parabolas
        v = np.empty(n, dtype=np.int64)

        # Separation points
        z = np.empty(n + 1, dtype=np.float64)

        k = 0

        v[0] = 0
        z[0] = -np.inf
        z[1] = np.inf

        for q in range(1, n):

            # Intersection between parabola q and parabola v[k]
            while True:

                vk = v[k]

                s = (
                    (f[q] + q * q)
                    - (f[vk] + vk * vk)
                ) / (2.0 * q - 2.0 * vk)

                if s <= z[k]:
                    k -= 1

                    if k < 0:
                        k = 0
                        break
                else:
                    break

            if k == 0 and v[0] == q:
                continue

            if q == 0:
                s = -np.inf

            v[k + 1] = q
            z[k + 1] = s
            z[k + 2] = np.inf

            k += 1

        k = 0

        for q in range(n):

            while z[k + 1] < q:
                k += 1

            vk = v[k]

            d[q] = (
                (q - vk) * (q - vk)
                + f[vk]
            )

        return d

    # --------------------------------------------------------
    # Transform along rows
    # --------------------------------------------------------

    tmp = np.empty_like(f)

    for y in range(h):
        tmp[y, :] = dt_1d(f[y, :])

    # --------------------------------------------------------
    # Transform along columns
    # --------------------------------------------------------

    out = np.empty_like(f)

    for x in range(w):
        out[:, x] = dt_1d(tmp[:, x])

    return np.sqrt(out)


# ============================================================
# 3. Construct disc
# ============================================================

def make_disc(radius):
    """
    Return integer offsets belonging to a circular disc.
    """

    r = int(radius)

    offsets = []

    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):

            if dx * dx + dy * dy <= r * r:
                offsets.append((dy, dx))

    return np.asarray(offsets, dtype=np.int32)


# ============================================================
# 4. Efficient full-disc summation
# ============================================================

def full_disc_sum(image, radius):
    """
    Compute the sum of image values under a full disc.

    This is NOT implemented using convolution.

    Instead, each row of the disc contributes a horizontal
    interval, and horizontal interval sums are obtained from
    prefix sums.

    Complexity is approximately O(H * W * radius).
    """

    h, w = image.shape[:2]

    # Number of channels
    if image.ndim == 2:
        channels = 1
        img = image[..., None]
    else:
        channels = image.shape[2]
        img = image

    img = img.astype(np.float64, copy=False)

    result = np.zeros(
        (h, w, channels),
        dtype=np.float64
    )

    r = int(radius)

    # For every vertical displacement, calculate the horizontal
    # width of the disc.
    for dy in range(-r, r + 1):

        dx_max = int(
            np.floor(
                np.sqrt(
                    r * r - dy * dy
                )
            )
        )

        # Shifted row
        y_src_start = max(0, -dy)
        y_src_end = min(h, h - dy)

        if y_src_start >= y_src_end:
            continue

        src = img[
            y_src_start:y_src_end
        ]

        # Prefix sum along x.
        # Extra zero column simplifies interval sums.
        prefix = np.zeros(
            (src.shape[0], w + 1, channels),
            dtype=np.float64
        )

        prefix[:, 1:] = np.cumsum(
            src,
            axis=1
        )

        # For output pixel x, interval is
        # [x-dx_max, x+dx_max].
        left = np.maximum(
            np.arange(w) - dx_max,
            0
        )

        right = np.minimum(
            np.arange(w) + dx_max + 1,
            w
        )

        row_sum = (
            prefix[:, right, :]
            - prefix[:, left, :]
        )

        result[
            y_src_start:y_src_end
        ] += row_sum

    if image.ndim == 2:
        return result[..., 0]

    return result


# ============================================================
# 5. Boundary-aware Bokeh filtering
# ============================================================

def bokeh_background(image, foreground_mask, diameter):
    """
    Blur only the background using a uniform disc-shaped
    Bokeh filter.

    At boundaries, the disc is cropped and renormalized.

    Foreground pixels remain unchanged.
    """

    image = image.astype(np.float64, copy=False)

    h, w = image.shape[:2]

    radius = diameter / 2.0

    # For integer pixel offsets, radius is 25 for D=50
    # and 50 for D=100.
    radius_int = int(radius)

    offsets = make_disc(radius_int)

    # --------------------------------------------------------
    # Background mask
    # --------------------------------------------------------

    background_mask = ~foreground_mask

    # --------------------------------------------------------
    # Distance from each background pixel to foreground.
    # --------------------------------------------------------

    # Distance to the nearest foreground pixel.
    dist_fg = distance_transform(foreground_mask)

    # Distance to image boundary.
    yy, xx = np.indices((h, w))

    dist_boundary = np.minimum.reduce([
        yy,
        xx,
        h - 1 - yy,
        w - 1 - xx
    ]).astype(np.float64)

    # A pixel is "interior" if the entire disc is:
    #
    #   1. inside the image
    #   2. inside the background
    #
    # Therefore no cropping/rescaling is required.
    interior = (
        background_mask
        & (dist_fg > radius)
        & (dist_boundary >= radius_int)
    )

    boundary = background_mask & ~interior

    # --------------------------------------------------------
    # Output starts as original image.
    # Foreground remains unchanged.
    # --------------------------------------------------------

    output = image.copy()

    # ========================================================
    # PART A:
    # Efficient processing of interior pixels
    # ========================================================

    # For interior pixels every point in the disc is background.
    # Hence the normalization factor is simply the number of
    # pixels in the disc.

    disc_area = len(offsets)

    disc_sum = full_disc_sum(
        image,
        radius_int
    )

    if image.ndim == 2:

        output[interior] = (
            disc_sum[interior]
            / disc_area
        )

    else:

        output[interior] = (
            disc_sum[interior]
            / disc_area
        )

    # ========================================================
    # PART B:
    # Boundary pixels
    # ========================================================

    # Here the disc can be cropped by:
    #
    #   - image boundary
    #   - foreground boundary
    #
    # We explicitly calculate:
    #
    #     sum(background pixels)
    #     ----------------------
    #     number of background pixels
    #
    # so the filter is automatically rescaled.

    boundary_positions = np.argwhere(boundary)

    if image.ndim == 2:

        for y, x in boundary_positions:

            total = 0.0
            count = 0

            for dy, dx in offsets:

                yy2 = y + dy
                xx2 = x + dx

                # Image boundary
                if (
                    yy2 < 0
                    or yy2 >= h
                    or xx2 < 0
                    or xx2 >= w
                ):
                    continue

                # Object boundary
                if not background_mask[yy2, xx2]:
                    continue

                total += image[yy2, xx2]
                count += 1

            if count > 0:
                output[y, x] = total / count

    else:

        for y, x in boundary_positions:

            total = np.zeros(
                image.shape[2],
                dtype=np.float64
            )

            count = 0

            for dy, dx in offsets:

                yy2 = y + dy
                xx2 = x + dx

                if (
                    yy2 < 0
                    or yy2 >= h
                    or xx2 < 0
                    or xx2 >= w
                ):
                    continue

                if not background_mask[yy2, xx2]:
                    continue

                total += image[yy2, xx2]
                count += 1

            if count > 0:
                output[y, x] = total / count

    return output


# ============================================================
# 6. Display
# ============================================================

def display_results(original, result50, result100, title):
    """
    Display original, D=50 and D=100 results using the
    same colorscale.
    """

    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 6)
    )

    # --------------------------------------------------------
    # RGB image
    # --------------------------------------------------------

    if original.ndim == 3:

        axes[0].imshow(
            np.clip(original, 0, 1)
        )

        axes[1].imshow(
            np.clip(result50, 0, 1)
        )

        axes[2].imshow(
            np.clip(result100, 0, 1)
        )

        axes[0].set_title("Original")
        axes[1].set_title("Bokeh, diameter = 50")
        axes[2].set_title("Bokeh, diameter = 100")

    # --------------------------------------------------------
    # Grayscale image
    # --------------------------------------------------------

    else:

        vmin = np.min(original)
        vmax = np.max(original)

        im0 = axes[0].imshow(
            original,
            cmap="gray",
            vmin=vmin,
            vmax=vmax
        )

        im1 = axes[1].imshow(
            result50,
            cmap="gray",
            vmin=vmin,
            vmax=vmax
        )

        im2 = axes[2].imshow(
            result100,
            cmap="gray",
            vmin=vmin,
            vmax=vmax
        )

        fig.colorbar(im0, ax=axes[0])
        fig.colorbar(im1, ax=axes[1])
        fig.colorbar(im2, ax=axes[2])

        axes[0].set_title("Original")
        axes[1].set_title("Bokeh, diameter = 50")
        axes[2].set_title("Bokeh, diameter = 100")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# ============================================================
# 7. Load images and run
# ============================================================

image_paths = [
    "data/bokeh/deep.png",
    "data/bokeh/lotus.png",
    "data/bokeh/marigold.png"
]

for path in image_paths:

    image = plt.imread(path)

    # Convert integer images to floating point.
    image = image.astype(np.float64)

    # If PNG is uint8-like and imread returns [0,255],
    # normalize it to [0,1].
    if np.max(image) > 1.0:
        image /= 255.0

    print("\n====================================")
    print("Image:", path)
    print("====================================")

    # --------------------------------------------------------
    # Manually select foreground
    # --------------------------------------------------------

    foreground_mask = select_foreground(
        image,
        "Select foreground: " + path
    )

    # --------------------------------------------------------
    # Show mask for verification
    # --------------------------------------------------------

    plt.figure(figsize=(8, 6))
    plt.imshow(foreground_mask, cmap="gray")
    plt.title("Foreground mask")
    plt.colorbar()
    plt.axis("off")
    plt.show()

    # --------------------------------------------------------
    # Diameter = 50
    # --------------------------------------------------------

    result50 = bokeh_background(
        image,
        foreground_mask,
        diameter=2
    )

    # --------------------------------------------------------
    # Diameter = 100
    # --------------------------------------------------------

    result100 = bokeh_background(
        image,
        foreground_mask,
        diameter=3
    )

    # --------------------------------------------------------
    # Display all three
    # --------------------------------------------------------

    display_results(
        image,
        result50,
        result100,
        path
    )