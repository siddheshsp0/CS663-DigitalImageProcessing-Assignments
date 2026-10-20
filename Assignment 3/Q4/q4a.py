import re
import zipfile
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


data_path = Path(__file__).resolve().parent.parent / "data" / "faces" / "ATT_Faces_Dataset.zip"
output_dir = Path(__file__).resolve().parent.parent / "output"

num_subjects = 40
images_per_subject = 10
image_shape = (112, 92)

# Training set: images 1-8 of subjects 1-30 (240 images)
train_subjects = 30
train_images = 8


def load_dataset(path):
    # faces[s, i] = image (i+1) of subject (s+1), flattened, intensities in [0, 1]
    faces = np.zeros((num_subjects, images_per_subject, image_shape[0] * image_shape[1]))

    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():

            # e.g. ATT_Faces_Dataset/s7/3.pgm or ATT_Faces_Dataset/s38/3_fl.pgm
            match = re.search(r"s(\d+)/(\d+)[^/]*\.pgm$", name)
            if match is None:
                continue

            subject, index = int(match[1]), int(match[2])

            buffer = np.frombuffer(archive.read(name), dtype=np.uint8)
            img = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)

            faces[subject - 1, index - 1] = img.astype(np.float64).ravel() / 255.0

    return faces


def training_set(faces):
    images = faces[:train_subjects, :train_images].reshape(-1, faces.shape[-1])
    labels = np.repeat(np.arange(1, train_subjects + 1), train_images)

    return images, labels


def compute_pca(images):
    # images: N x d, one flattened image per row
    mean = images.mean(axis=0)

    X = (images - mean).T          # d x N, columns are mean-subtracted images
    N = X.shape[1]

    # Covariance C = X X^T / N is d x d (10304 x 10304). If w is an eigenvector
    # of the small N x N matrix X^T X / N, then X w is an eigenvector of C with
    # the same eigenvalue, so we decompose the small matrix instead.
    eigenvalues, w = np.linalg.eigh(X.T @ X / N)

    # Sort in decreasing order of variance; the mean-subtracted data has rank
    # N - 1, so the last eigenvalue is 0 and is dropped
    eigenvalues = eigenvalues[::-1][:N - 1]
    w = w[:, ::-1][:, :N - 1]

    eigenvectors = X @ w
    eigenvectors /= np.linalg.norm(eigenvectors, axis=0)

    # eigenvectors: d x (N - 1), unit-norm columns = eigenfaces
    return mean, eigenvalues, eigenvectors


if __name__ == "__main__":

    output_dir.mkdir(exist_ok=True)

    faces = load_dataset(data_path)
    train, _ = training_set(faces)

    mean, eigenvalues, eigenvectors = compute_pca(train)

    # Mean image
    fig, ax = plt.subplots(figsize=(5, 6))

    im = ax.imshow(mean.reshape(image_shape), cmap="gray")
    ax.set_title("Mean face (240 training images)")
    ax.axis("off")
    fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(output_dir / "q4a1.png", dpi=150, bbox_inches="tight")
    plt.show()

    # First 10 principal modes of variation: mu +- alpha v_i, alpha = k sqrt(lambda_i)
    num_modes = 10
    k = 2

    eigenfaces = [eigenvectors[:, i].reshape(image_shape) for i in range(num_modes)]
    alphas = [k * np.sqrt(eigenvalues[i]) for i in range(num_modes)]

    minus = [mean.reshape(image_shape) - a * v for a, v in zip(alphas, eigenfaces)]
    plus = [mean.reshape(image_shape) + a * v for a, v in zip(alphas, eigenfaces)]

    # Common colour range per row so that the images can be compared
    v_max = max(np.abs(v).max() for v in eigenfaces)
    face_min = min(img.min() for img in minus + plus)
    face_max = max(img.max() for img in minus + plus)

    rows = [
        (eigenfaces, "eigenface $v_i$", -v_max, v_max),
        (minus, rf"$\mu - {k}\sqrt{{\lambda_i}}\, v_i$", face_min, face_max),
        (plus, rf"$\mu + {k}\sqrt{{\lambda_i}}\, v_i$", face_min, face_max),
    ]

    fig, ax = plt.subplots(3, num_modes, figsize=(22, 8.4))

    for row, (images, label, v_lo, v_hi) in enumerate(rows):
        for i in range(num_modes):

            im = ax[row, i].imshow(images[i], cmap="gray", vmin=v_lo, vmax=v_hi)
            ax[row, i].set_xticks([])
            ax[row, i].set_yticks([])

            if row == 0:
                ax[row, i].set_title(f"Mode {i + 1}\n$\\lambda$={eigenvalues[i]:.2f}")

        ax[row, 0].set_ylabel(label, fontsize=13)
        fig.colorbar(im, ax=ax[row, :], fraction=0.015, pad=0.01)

    fig.suptitle("First 10 principal modes of variation", fontsize=16)
    plt.savefig(output_dir / "q4a2.png", dpi=150, bbox_inches="tight")
    plt.show()
