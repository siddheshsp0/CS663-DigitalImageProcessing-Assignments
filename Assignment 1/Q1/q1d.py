import numpy as np
import cv2
import matplotlib.pyplot as plt

IMG_PATH='./data/interp/random.png'


def myBicubicInterpolation(img: np.ndarray):
    M, N = img.shape
    M_new = 300 * (M - 1) + 1
    N_new = 300 * (N - 1) + 1

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
        y = i / 300.0
        y1 = int(np.floor(y))
        y2 = min(y1 + 1, M - 1)

        # Local coordinate inside the current unit square
        v = y - y1
        for j in range(N_new):
            # Coordinate in original image
            x = j / 300.0

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





def main():
    img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)

    img_ = myBicubicInterpolation(img)
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
