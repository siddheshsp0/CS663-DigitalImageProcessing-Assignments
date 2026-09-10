import cv2
import numpy as np
import matplotlib.pyplot as plt

IMG_PATH='./data/interp/suit.png' # Image path


def myImageShrink(img: np.ndarray, factor: int):
    shape = img.shape
    output = np.zeros((
        int(shape[0] / factor),
        int(shape[1] / factor),
        shape[2]
    ), dtype=img.dtype)

    for i in range(int(shape[0] / factor)):
        for j in range(int(shape[1] / factor)):
            output[i][j] = img[i * factor][j * factor]

    return output


def main():
    # Read image
    img = cv2.imread(IMG_PATH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Subsample fnction call
    imgd2 = myImageShrink(img, 2)
    imgd3 = myImageShrink(img, 2)

    # Create 3 plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # Original
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal")
    # d = 2
    axes[1].imshow(imgd2)
    axes[1].set_title("Subsampled (d = 2)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_aspect("equal")
    # d = 3
    axes[2].imshow(imgd3)
    axes[2].set_title("Subsampled (d = 3)")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_aspect("equal")

    plt.tight_layout()
    plt.show()
    


if __name__=='__main__':
    main()
