import cv2
import numpy as np
import matplotlib.pyplot as plt

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



angle = 5.0
IMG_PATH = "data/interp/main.png"

def myImageRotationUsingNearestNeighborInterp(img, angle):
    M, N = img.shape[:2]
    output = np.zeros_like(img)
    theta = np.deg2rad(angle)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Center image
    cx = (N - 1) / 2.0
    cy = (M - 1) / 2.0
    for i in range(M):
        for j in range(N):

            x = (cx + cos_theta * (j - cx) + sin_theta * (i - cy))

            y = (cy - sin_theta * (j - cx) + cos_theta * (i - cy))

            # ifOutside original image
            if x < 0 or x > N - 1 or y < 0 or y > M - 1:
                continue
#NN interpolation

            x_nn = int(round(x))
            y_nn = int(round(y))
            x_nn = min(max(x_nn, 0), N - 1)
            y_nn = min(max(y_nn, 0), M - 1)

            output[i, j] = img[y_nn, x_nn]

    return output


def myImageRotationUsingBilinearInterp(img, angle):
    M, N = img.shape[:2]

    output = np.zeros_like(img, dtype=np.float64)
    theta = np.deg2rad(angle)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Center image
    cx = (N - 1) / 2.0
    cy = (M - 1) / 2.0
    for i in range(M):
        for j in range(N):
            x = (cx + cos_theta * (j - cx) + sin_theta * (i - cy))

            y = (cy - sin_theta * (j - cx) + cos_theta * (i - cy))

            # Outside original image
            if x < 0 or x > N - 1 or y < 0 or y > M - 1:
                continue

            # 4 neighbouring pixels

            x1 = int(np.floor(x))
            x2 = min(x1 + 1, N - 1)

            y1 = int(np.floor(y))
            y2 = min(y1 + 1, M - 1)

            # Fractional coordinates
            alpha = x - x1
            beta = y - y1

            Q11 = img[y1, x1]
            Q21 = img[y1, x2]
            Q12 = img[y2, x1]
            Q22 = img[y2, x2]

            #interpolate along x

            R1 = (1 - alpha) * Q11 + alpha * Q21
            R2 = (1 - alpha) * Q12 + alpha * Q22

            # interpolate along y
            value = (1 - beta) * R1 + beta * R2

            output[i, j] = value

    return output






if __name__ == '__main__':
    img = cv2.imread(
        IMG_PATH,
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        raise FileNotFoundError(
            "Could not read data/interp/main.png"
        )

    rotated_nn = myImageRotationUsingNearestNeighborInterp(img,angle)
    rotated_bilinear = myImageRotationUsingBilinearInterp(img,angle)

    fig, axes = plt.subplots(1, 3,figsize=(18, 6))

    # Original
    axes[0].imshow(img,cmap="gray",aspect="equal")
    axes[0].set_title("Original")
    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("Row")

    # Nearest neighbor
    axes[1].imshow(rotated_nn,cmap="gray",aspect="equal")
    axes[1].set_title(f"Nearest Neighbor ({angle}°)")
    axes[1].set_xlabel("Column")
    axes[1].set_ylabel("Row")

    # Bilinear
    axes[2].imshow(rotated_bilinear,cmap="gray",aspect="equal")
    axes[2].set_title(f"Bilinear ({angle}°)")
    axes[2].set_xlabel("Column")
    axes[2].set_ylabel("Row")


    plt.tight_layout()
    plt.show()