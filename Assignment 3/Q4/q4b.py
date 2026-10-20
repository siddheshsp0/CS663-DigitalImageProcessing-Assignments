import numpy as np

from q4a import (
    load_dataset, training_set, compute_pca, data_path,
    num_subjects, images_per_subject,
)


def distance_from_face_space(images, mean, eigenvectors):
    # Project the mean-subtracted image onto the face space (spanned by the
    # eigenfaces), reconstruct it, and measure what is left over:
    # score = || (x - mu) - U U^T (x - mu) ||
    centred = images - mean

    weights = centred @ eigenvectors
    reconstruction = weights @ eigenvectors.T

    return np.linalg.norm(centred - reconstruction, axis=1)


def compute_M1(faces, mean, eigenvectors):
    # M1[s, i] = distance of image (i+1) of subject (s+1) from the face space
    images = faces.reshape(-1, faces.shape[-1])

    scores = distance_from_face_space(images, mean, eigenvectors)

    return scores.reshape(num_subjects, images_per_subject)


if __name__ == "__main__":

    from q4d import tuned_params

    faces = load_dataset(data_path)
    train, _ = training_set(faces)

    mean, eigenvalues, eigenvectors = compute_pca(train)

    k = tuned_params["num_eigenfaces"]
    M1 = compute_M1(faces, mean, eigenvectors[:, :k])

    np.set_printoptions(precision=2, suppress=True, linewidth=150)
    print(f"M1 (distance from face space, {k} eigenfaces), rows = subjects, cols = images:")
    print(M1)
