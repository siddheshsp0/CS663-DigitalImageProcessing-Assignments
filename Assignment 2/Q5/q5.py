import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path


def select_foreground(image, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
    ax.set_title(title + "\nLeft click: add point | Right click: finish")

    points = []

    def onclick(event):
        if event.inaxes != ax:
            return

        if event.button == 1 and event.xdata is not None and event.ydata is not None:
            points.append((event.xdata, event.ydata))
            ax.plot(event.xdata, event.ydata, "ro", markersize=4)

            if len(points) > 1:
                x = [points[-2][0], points[-1][0]]
                y = [points[-2][1], points[-1][1]]
                ax.plot(x, y, "r-")

            fig.canvas.draw_idle()

        elif event.button == 3:
            if len(points) < 3:
                print("Need at least 3 points.")
                return

            x = [p[0] for p in points] + [points[0][0]]
            y = [p[1] for p in points] + [points[0][1]]
            ax.plot(x, y, "r-")
            fig.canvas.draw_idle()

            fig.canvas.mpl_disconnect(cid)
            plt.close(fig)

    cid = fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()

    if len(points) < 3:
        raise ValueError("No valid polygon was selected.")

    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    coordinates = np.column_stack((xx.ravel(), yy.ravel()))

    return Path(points).contains_points(coordinates).reshape(h, w)


def distance_transform(binary):
    h, w = binary.shape
    INF = 1e12
    f = np.where(binary, 0.0, INF)

    def dt_1d(f):
        n = len(f)
        d = np.empty(n, dtype=np.float64)

        v = np.empty(n, dtype=np.int64)
        z = np.empty(n + 1, dtype=np.float64)

        k = 0
        v[0] = 0
        z[0] = -np.inf
        z[1] = np.inf

        for q in range(1, n):
            while True:
                vk = v[k]
                s = ((f[q] + q * q) - (f[vk] + vk * vk)) / (
                    2.0 * q - 2.0 * vk
                )

                if s <= z[k]:
                    k -= 1
                else:
                    break

                if k < 0:
                    k = 0
                    break

            k += 1
            v[k] = q
            z[k] = s
            z[k + 1] = np.inf

        k = 0
        for q in range(n):
            while z[k + 1] < q:
                k += 1

            vk = v[k]
            d[q] = (q - vk) ** 2 + f[vk]

        return d

    tmp = np.empty_like(f)

    for y in range(h):
        tmp[y] = dt_1d(f[y])

    out = np.empty_like(f)

    for x in range(w):
        out[:, x] = dt_1d(tmp[:, x])

    return np.sqrt(out)


def disc_row_spans(radius):
    r = int(radius)
    spans = []

    for dy in range(-r, r + 1):
        dx = int(np.floor(np.sqrt(r * r - dy * dy)))
        spans.append((dy, dx))

    return spans


def integral_image(image):
    if image.ndim == 2:
        image = image[..., None]

    h, w, c = image.shape

    integral = np.zeros((h + 1, w + 1, c), dtype=np.float64)
    integral[1:, 1:] = np.cumsum(
        np.cumsum(image, axis=0),
        axis=1
    )

    return integral


def rectangle_sum(integral, y1, y2, x1, x2):
    h = integral.shape[0] - 1
    w = integral.shape[1] - 1

    y1 = np.clip(y1, 0, h - 1)
    y2 = np.clip(y2, 0, h - 1)
    x1 = np.clip(x1, 0, w - 1)
    x2 = np.clip(x2, 0, w - 1)

    return (
        integral[y2 + 1, x2 + 1]
        - integral[y1, x2 + 1]
        - integral[y2 + 1, x1]
        + integral[y1, x1]
    )


def disc_sum_all_pixels(integral, radius):
    h = integral.shape[0] - 1
    w = integral.shape[1] - 1
    c = integral.shape[2]

    result = np.zeros((h, w, c), dtype=np.float64)
    x = np.arange(w)

    for dy, dx in disc_row_spans(radius):
        y_out = np.arange(h)
        y_src = y_out + dy

        valid_y = (y_src >= 0) & (y_src < h)

        if not np.any(valid_y):
            continue

        ys = y_src[valid_y]

        x1 = np.maximum(x - dx, 0)
        x2 = np.minimum(x + dx, w - 1)

        row_sum = (
            integral[ys[:, None] + 1, x2[None, :] + 1]
            - integral[ys[:, None], x2[None, :] + 1]
            - integral[ys[:, None] + 1, x1[None, :]]
            + integral[ys[:, None], x1[None, :]]
        )

        result[valid_y] += row_sum

    return result


def disc_sum_at_pixels(integral, ys, xs, radius):
    c = integral.shape[2]
    result = np.zeros((len(ys), c), dtype=np.float64)

    for dy, dx in disc_row_spans(radius):
        y = ys + dy

        valid = (y >= 0) & (y < integral.shape[0] - 1)

        if not np.any(valid):
            continue

        xv = xs[valid]
        yv = y[valid]

        x1 = np.maximum(xv - dx, 0)
        x2 = np.minimum(xv + dx, integral.shape[1] - 2)

        row_sum = (
            integral[yv + 1, x2 + 1]
            - integral[yv, x2 + 1]
            - integral[yv + 1, x1]
            + integral[yv, x1]
        )

        result[valid] += row_sum

    return result


def bokeh_background(image, foreground_mask, diameter):
    image = np.asarray(image, dtype=np.float64)

    if image.ndim == 2:
        img = image[..., None]
        grayscale = True
    else:
        img = image
        grayscale = False

    h, w, c = img.shape
    radius = int(diameter // 2)

    background = ~foreground_mask

    dist_fg = distance_transform(foreground_mask)

    yy, xx = np.indices((h, w))
    dist_edge = np.minimum.reduce([
        yy,
        xx,
        h - 1 - yy,
        w - 1 - xx
    ])

    interior = (
        background
        & (dist_fg > radius)
        & (dist_edge >= radius)
    )

    boundary = background & ~interior
    output = img.copy()

    bg_img = img * background[..., None]
    bg_count = background.astype(np.float64)[..., None]

    integral_img = integral_image(bg_img)
    integral_count = integral_image(bg_count)

    disc_area = sum(
        2 * dx + 1
        for _, dx in disc_row_spans(radius)
    )

    if np.any(interior):
        disc_sums = disc_sum_all_pixels(integral_img, radius)
        output[interior] = disc_sums[interior] / disc_area

    if np.any(boundary):
        ys, xs = np.nonzero(boundary)

        sums = disc_sum_at_pixels(
            integral_img,
            ys,
            xs,
            radius
        )

        counts = disc_sum_at_pixels(
            integral_count,
            ys,
            xs,
            radius
        )[:, 0]

        valid = counts > 0
        output[ys[valid], xs[valid]] = (
            sums[valid] / counts[valid, None]
        )

    if grayscale:
        return output[..., 0]

    return output


def display_results(original, result50, result100, title):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    if original.ndim == 3:
        axes[0].imshow(np.clip(original, 0, 1))
        axes[1].imshow(np.clip(result50, 0, 1))
        axes[2].imshow(np.clip(result100, 0, 1))
    else:
        vmin = np.min(original)
        vmax = np.max(original)

        axes[0].imshow(
            original,
            cmap="gray",
            vmin=vmin,
            vmax=vmax
        )
        axes[1].imshow(
            result50,
            cmap="gray",
            vmin=vmin,
            vmax=vmax
        )
        axes[2].imshow(
            result100,
            cmap="gray",
            vmin=vmin,
            vmax=vmax
        )

    axes[0].set_title("Original")
    axes[1].set_title("Bokeh, diameter = 50")
    axes[2].set_title("Bokeh, diameter = 100")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


image_paths = [
    "data/bokeh/marigold.png",
    "data/bokeh/deep.png",
    "data/bokeh/lotus.png",
]

for path in image_paths:
    image = plt.imread(path).astype(np.float64)

    if np.max(image) > 1.0:
        image /= 255.0

    print("\nImage:", path)

    foreground_mask = select_foreground(
        image,
        "Select foreground: " + path
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(
        foreground_mask,
        cmap="gray",
        vmin=0,
        vmax=1
    )
    plt.title("Foreground mask")
    plt.axis("off")
    plt.show()

    result50 = bokeh_background(
        image,
        foreground_mask,
        diameter=50
    )

    result100 = bokeh_background(
        image,
        foreground_mask,
        diameter=100
    )

    display_results(
        image,
        result50,
        result100,
        path
    )