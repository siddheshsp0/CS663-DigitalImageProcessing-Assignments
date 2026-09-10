import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


def gaussian_kernel(size, sigma):

    r = size // 2
    x = np.arange(-r, r + 1)
    y = np.arange(-r, r + 1)
    
    X, Y = np.meshgrid(x, y)
    
    kernel = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)

    return kernel


def convolve2d(image, kernel):
    h, w = image.shape

    k = kernel.shape[0]
    r = k // 2

    padded = np.pad(image, r, mode="edge")

    output = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):

            region = padded[i:i+k, j:j+k]

            output[i, j] = np.sum(region * kernel)

    return output


def gaussian_blur_rgb(image, kernel):
    h, w, c = image.shape

    blurred = np.zeros((h, w, c), dtype=np.float32)

    for channel in range(3):
        blurred[:, :, channel] = convolve2d( image[:, :, channel], kernel)

    return blurred

def unsharp_mask(img, sigma=2.0, s=1.0):
    
    img_float = img.astype(np.float32)

    blurred = gaussian_blur_rgb(img_float, gaussian_kernel(11, sigma))

    mask = img_float - blurred
    sharp = img_float + s * mask

    # Clip to input image intensity range
    sharp = np.clip(
        sharp,
        img_float.min(),
        img_float.max()
    )

    return sharp.astype(np.uint8)


data_dir = Path(__file__).resolve().parent.parent / "data" / "sharpen"
image_paths = [data_dir / "moon.png",data_dir / "peacock.png",data_dir / "tiger.png"]

s1 = 1.0
s2 = 3.0
sigma = 2.0

for path in image_paths:

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    sharp1 = unsharp_mask(img, sigma=sigma, s=s1)
    sharp2 = unsharp_mask(img, sigma=sigma, s=s2)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(img)
    ax[0].set_title("Original")

    ax[1].imshow(sharp1)
    ax[1].set_title(f"Sharpened (s={s1})")

    ax[2].imshow(sharp2)
    ax[2].set_title(f"Sharpened (s={s2})")

    for a in ax:
        a.axis("off")

    plt.suptitle(path.name)
    plt.tight_layout()
    plt.show()