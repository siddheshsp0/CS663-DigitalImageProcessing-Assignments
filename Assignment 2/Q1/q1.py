import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def unsharp_mask(img, sigma=2.0, s=1.0):
    
    img_float = img.astype(np.float32)

    blurred = cv2.GaussianBlur( img_float, (0, 0), sigmaX=sigma, sigmaY=sigma)

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