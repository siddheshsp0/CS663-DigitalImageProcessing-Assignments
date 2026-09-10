import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


def myNearestNeighborInterpolation(img: np.ndarray, M_new: int, N_new: int):
    M, N = img.shape
    output = np.zeros((M_new, N_new), dtype=img.dtype)
    for i in range(M_new):
        for j in range(N_new):
            y = i * (M - 1) / (M_new - 1)
            x = j * (N - 1) / (N_new - 1)

            y_nearest = min(int(np.floor(y + 0.5)), M - 1)
            x_nearest = min(int(np.floor(x + 0.5)), N - 1)
            output[i, j] = img[x_nearest, y_nearest]
    return output

def myBilinearInterpolation(img: np.ndarray, M_new: int, N_new: int):
    M, N = img.shape
    output = np.zeros((M_new, N_new), dtype=np.float64)
    for i in range(M_new):
        # y coordinate in original image
        y = i * (M - 1) / (M_new - 1)
        y0 = int(np.floor(y))
        y1 = min(y0 + 1, M - 1)
        beta = y - y0

        for j in range(N_new):
            # x coordinate in original image
            x = j * (N - 1) / (N_new - 1)

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


def myBicubicInterpolation(img: np.ndarray, M_new: int, N_new: int):
    M, N = img.shape

    output = np.zeros((M_new, N_new), dtype=np.float64)

    # Computing fx, fy and fxy at every pixel using finite differences
    fx = np.zeros((M, N), dtype=np.float64)
    fy = np.zeros((M, N), dtype=np.float64)
    fxy = np.zeros((M, N), dtype=np.float64)

    # fx
    # interior
    fx[:, 1:-1] = 0.5 * (img[:, 2:] - img[:, :-2])
    # boundary
    fx[:, 0] = img[:, 1] - img[:, 0]
    fx[:, -1] = img[:, -1] - img[:, -2]

    # fy
    # interior
    fy[1:-1, :] = 0.5 * (img[2:, :] - img[:-2, :])
    # boundary
    fy[0, :] = img[1, :] - img[0, :]
    fy[-1, :] = img[-1, :] - img[-2, :]

    # fxy
    # interior
    fxy[1:-1, 1:-1] = 0.25 * (
        img[2:, 2:]
        + img[:-2, :-2]
        - img[:-2, 2:]
        - img[2:, :-2]
    )
    # boundary
    fxy[:, 0] = fy[:, 1] - fy[:, 0]
    fxy[:, -1] = fy[:, -1] - fy[:, -2]


    # system of equations to solve this bicubic problem:
    A = np.zeros((16, 16), dtype=np.float64)
    row = 0
    corners = [
        (0, 0),   # Q11
        (1, 0),   # Q21
        (0, 1),   # Q12
        (1, 1)    # Q22
    ]

    # Helper functions for the basis
    def basis_p(x, y):
        # Coefficients of a_ij in p(x,y)
        return np.array([
            x**i * y**j
            for i in range(4)
            for j in range(4)
        ])

    def basis_px(x, y):
        # Coefficients of a_ij in p_x(x,y)
        return np.array([
            (i * x**(i - 1) * y**j) if i >= 1 else 0.0
            for i in range(4)
            for j in range(4)
        ])

    def basis_py(x, y):
        # Coefficients of a_ij in p_y(x,y)
        return np.array([
            (j * x**i * y**(j - 1)) if j >= 1 else 0.0
            for i in range(4)
            for j in range(4)
        ])

    def basis_pxy(x, y):
        #Coefficients of a_ij in p_xy(x,y)
        return np.array([
            (i * j * x**(i - 1) * y**(j - 1))
            if i >= 1 and j >= 1 else 0.0
            for i in range(4)
            for j in range(4)
        ])

    # Build the 16 equations
    for x, y in corners:
        # p(x,y) = f(x,y)
        A[row, :] = basis_p(x, y)
        row += 1
        # px(x,y) = fx(x,y)
        A[row, :] = basis_px(x, y)
        row += 1
        # py(x,y) = fy(x,y)
        A[row, :] = basis_py(x, y)
        row += 1
        # pxy(x,y) = fxy(x,y)
        A[row, :] = basis_pxy(x, y)
        row += 1

    # computing A inverse
    A_inv = np.linalg.inv(A)

    #interpolate outer pixels
    for i in range(M_new):
        # Coordinate in original image
        y = i * (M - 1) / (M_new - 1)
        y1 = int(np.floor(y))
        y2 = min(y1 + 1, M - 1)

        # Local coordinate inside the current unit square
        v = y - y1
        for j in range(N_new):
            # Coordinate in original image
            x = j * (N - 1) / (N_new - 1)

            x1 = int(np.floor(x))
            x2 = min(x1 + 1, N - 1)
            # Local coordinate inside current unit square
            u = x - x1

            # Data at the four corners
            # Q11 = (x1,y1)
            # Q21 = (x2,y1)
            # Q12 = (x1,y2)
            # Q22 = (x2,y2)
            b = np.array([
                img[y1, x1],
                fx[y1, x1],
                fy[y1, x1],
                fxy[y1, x1],

                img[y1, x2],
                fx[y1, x2],
                fy[y1, x2],
                fxy[y1, x2],

                img[y2, x1],
                fx[y2, x1],
                fy[y2, x1],
                fxy[y2, x1],

                img[y2, x2],
                fx[y2, x2],
                fy[y2, x2],
                fxy[y2, x2]
            ], dtype=np.float64)

            # Solve the 16 equations for the 16 coefficients
            # A@coefficients = b

            coefficients = A_inv @ b
            # final pixel value
            value = 0.0
            k = 0
            for ii in range(4):
                for jj in range(4):

                    value += (
                        coefficients[k]
                        * u**ii
                        * v**jj
                    )

                    k += 1

            output[i, j] = value

    return output


if __name__ == "__main__":
    # Load MATLAB file
    data = loadmat("data/interp/ct.mat")
    print("Variables in ct.mat:")
    print(data.keys())

    # Extract images
    original = np.asarray(
        data["original"],
        dtype=np.float64
    )
    subsampled = np.asarray(
        data["subsampled"],
        dtype=np.float64
    )
    print("Original shape:", original.shape)
    print("Subsampled shape:", subsampled.shape)

    M_original, N_original = original.shape

#interpolation
    enlarged_nn = myNearestNeighborInterpolation(
        subsampled,
        M_original,
        N_original
    )

    enlarged_bilinear = myBilinearInterpolation(
        subsampled,
        M_original,
        N_original
    )

    enlarged_bicubic = myBicubicInterpolation(
        subsampled,
        M_original,
        N_original
    )

    # convert all results to float
    enlarged_nn = enlarged_nn.astype(np.float64)

    # Difference images
    difference_nn = original - enlarged_nn
    difference_bilinear = (
        original - enlarged_bilinear
    )
    difference_bicubic = (
        original - enlarged_bicubic
    )

    # RMSE
    rmse_nn = np.sqrt(
        np.mean(
            (original - enlarged_nn) ** 2
        )
    )
    rmse_bilinear = np.sqrt(
        np.mean(
            (original - enlarged_bilinear) ** 2
        )
    )
    rmse_bicubic = np.sqrt(
        np.mean(
            (original - enlarged_bicubic) ** 2
        )
    )

    print("\n================ RMSE ================\n")

    print(
        f"Nearest Neighbor : {rmse_nn:.6f}"
    )
    print(
        f"Bilinear         : {rmse_bilinear:.6f}"
    )
    print(
        f"Bicubic          : {rmse_bicubic:.6f}"
    )

    # Common color limits for all original/enlarged images (asked in doc)
    image_vmin = min(original.min(),enlarged_nn.min(),enlarged_bilinear.min(),enlarged_bicubic.min())
    image_vmax = max(original.max(),enlarged_nn.max(),enlarged_bilinear.max(),enlarged_bicubic.max())


    # Common color limits for ALL difference images
    # Symmetric around zero

    diff_max = max(np.abs(difference_nn).max(),np.abs(difference_bilinear).max(),np.abs(difference_bicubic).max())
    diff_vmin = -diff_max
    diff_vmax = diff_max

    # Figure 1: Original + three interpolated images

    fig1, axes = plt.subplots(1,4,figsize=(20, 5))
    im = axes[0].imshow(original,cmap="jet",vmin=image_vmin,vmax=image_vmax,aspect="equal")

    axes[0].set_title("Original")
    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("Row")

    axes[1].imshow(enlarged_nn,cmap="jet",vmin=image_vmin,vmax=image_vmax,aspect="equal")

    axes[1].set_title(f"Nearest Neighbor\nRMSE = {rmse_nn:.4f}")
    axes[1].set_xlabel("Column")
    axes[1].set_ylabel("Row")

    axes[2].imshow(enlarged_bilinear,cmap="jet",vmin=image_vmin,vmax=image_vmax,aspect="equal")

    axes[2].set_title(f"Bilinear\nRMSE = {rmse_bilinear:.4f}")
    axes[2].set_xlabel("Column")
    axes[2].set_ylabel("Row")

    axes[3].imshow(enlarged_bicubic,cmap="jet",vmin=image_vmin,vmax=image_vmax,aspect="equal")

    axes[3].set_title(f"Bicubic\nRMSE = {rmse_bicubic:.4f}")
    axes[3].set_xlabel("Column")
    axes[3].set_ylabel("Row")

    # One common colorbar
    fig1.colorbar(im,ax=axes,shrink=0.8,label="Intensity")

    fig1.suptitle("CT Image: Original vs Interpolated Images",fontsize=16)

    plt.tight_layout()
    plt.show()

    # Figure 2: Difference images
    fig2, axes = plt.subplots(1,3,figsize=(16, 5))

    im_diff = axes[0].imshow(difference_nn,cmap="jet",vmin=diff_vmin,vmax=diff_vmax,aspect="equal")

    axes[0].set_title("Original - Nearest Neighbor")
    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("Row")

    axes[1].imshow(difference_bilinear,cmap="jet",vmin=diff_vmin,vmax=diff_vmax,aspect="equal")

    axes[1].set_title(
        "Original - Bilinear"
    )
    axes[1].set_xlabel("Column")
    axes[1].set_ylabel("Row")

    axes[2].imshow(difference_bicubic,cmap="jet",vmin=diff_vmin,vmax=diff_vmax,aspect="equal"
    )

    axes[2].set_title("Original - Bicubic")
    axes[2].set_xlabel("Column")
    axes[2].set_ylabel("Row")

    # common colorbar for all difference images
    fig2.colorbar(im_diff,ax=axes,shrink=0.8,label="Original - Interpolated")

    fig2.suptitle("Interpolation Difference Images",fontsize=16)

    plt.tight_layout()
    plt.show()