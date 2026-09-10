import numpy as np
import matplotlib.pyplot as plt
import cv2

IMG_PATH = "data/interp/random.png"

def myBilinearInterpolation(img: np.ndarray):
    M, N = img.shape
    M_new = 300 * (M - 1) + 1
    N_new = 300 * (N - 1) + 1

    output = np.zeros((M_new, N_new), dtype=np.float64)
    for i in range(M_new):
        # y coordinate in original image
        y = i / 300.0
        y0 = int(np.floor(y))
        y1 = min(y0 + 1, M - 1)
        beta = y - y0

        for j in range(N_new):
            # x coordinate in original image
            x = j / 300.0

            x0 = int(np.floor(x))
            x1 = min(x0 + 1, N - 1)
            alpha = x - x0

            # 4 neighboring pixels
            I00 = img[y0, x0]
            I01 = img[y0, x1]
            I10 = img[y1, x0]
            I11 = img[y1, x1]

            # bilinear interpolation
            top = (1 - alpha) * I00 + alpha * I01
            bottom = (1 - alpha) * I10 + alpha * I11

            output[i, j] = (1 - beta) * top + beta * bottom

    return output




def main():
    img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    img_ = myBilinearInterpolation(img)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Original
    im1 = axes[0].imshow(img, cmap="jet")
    axes[0].set_title("Original")
    axes[0].set_xlabel("x (pixels)")
    axes[0].set_ylabel("y (pixels)")
    axes[0].set_aspect("equal")
    fig.colorbar(im1, ax=axes[0])
    # Interpolated
    im2 = axes[1].imshow(img_, cmap="jet")
    axes[1].set_title("Nearest-Neighbor Interpolated")
    axes[1].set_xlabel("x (pixels)")
    axes[1].set_ylabel("y (pixels)")
    axes[1].set_aspect("equal")
    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()