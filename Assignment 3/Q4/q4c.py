import numpy as np

from q4a import (
    load_dataset, training_set, compute_pca, data_path,
    num_subjects, images_per_subject,
)


def recognize(images, mean, eigenvectors, train_weights, train_labels):
    # Face-space projection (weight vector) of each given image
    weights = (images - mean) @ eigenvectors

    # Distance to the face-space projection of every training image;
    # the nearest one gives the identity of the person
    distances = np.linalg.norm(
        weights[:, None, :] - train_weights[None, :, :], axis=2
    )
    nearest = np.argmin(distances, axis=1)

    return distances[np.arange(len(images)), nearest], train_labels[nearest]


def compute_M2(faces, mean, eigenvectors, train, train_labels):
    # M2[s, i] = distance between the face-space projection of image (i+1) of
    # subject (s+1) and its nearest training-image projection.
    # Also returns the identity of that nearest training image.
    images = faces.reshape(-1, faces.shape[-1])
    train_weights = (train - mean) @ eigenvectors

    distances, identities = recognize(
        images, mean, eigenvectors, train_weights, train_labels
    )

    shape = (num_subjects, images_per_subject)
    return distances.reshape(shape), identities.reshape(shape)


if __name__ == "__main__":

    from q4d import tuned_params

    faces = load_dataset(data_path)
    train, train_labels = training_set(faces)

    mean, eigenvalues, eigenvectors = compute_pca(train)

    k = tuned_params["num_eigenfaces"]
    M2, identities = compute_M2(faces, mean, eigenvectors[:, :k], train, train_labels)

    np.set_printoptions(precision=2, suppress=True, linewidth=150)
    print(f"M2 (distance to nearest training face in face space, {k} eigenfaces):")
    print(M2)
    print("\nIdentity of the nearest training face:")
    print(identities)
