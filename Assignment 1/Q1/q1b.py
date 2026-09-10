# q1a.py
import cv2
import numpy as np
import matplotlib.pyplot as plt


IMG_PATH = "./data/interp/random.png"


def myNearestNeighborInterpolation(img: np.ndarray):
    M, N = img.shape
    M_new = 300 * (M - 1) + 1
    N_new = 300 * (N - 1) + 1
    output = np.zeros((M_new, N_new), dtype=img.dtype)
    for i in range(M_new):
        for j in range(N_new):
            x = i / 300
            y = j / 300
            x_nearest = int(np.floor(x + 0.5))
            y_nearest = int(np.floor(y + 0.5))
            output[i, j] = img[x_nearest, y_nearest]
    return output


def main():
    img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)

    img_ = myNearestNeighborInterpolation(img)
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

if __name__=='__main__':
    main()
