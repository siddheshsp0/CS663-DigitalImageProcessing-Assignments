import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle

from q4a import (
    load_dataset, training_set, compute_pca, data_path, output_dir,
    num_subjects, images_per_subject, train_subjects, train_images,
)
from q4b import compute_M1
from q4c import compute_M2


# Subjects 31-37 are faces that are absent from the training set.
# Subjects 38-40 are not face images: s38 = upside-down faces,
# s39 = faces buried in salt & pepper noise, s40 = intensity-negated faces.
known_subjects = 30
face_subjects = 37

# Free parameters, tuned on the test set by tune_parameters() below
tuned_params = dict(num_eigenfaces=26, face_threshold=12.845, known_threshold=6.722)


def decide(M1, M2, identities, face_threshold, known_threshold):
    # -1    : too far from the face space            -> not a face
    #  0    : face, but far from every training face -> person not in training set
    #  index: identity of the nearest training face
    return np.where(
        M1 > face_threshold, -1,
        np.where(M2 > known_threshold, 0, identities)
    )


def ground_truth():
    truth = np.zeros((num_subjects, images_per_subject), dtype=int)

    truth[:known_subjects] = np.arange(1, known_subjects + 1)[:, None]
    truth[face_subjects:] = -1

    return truth


def test_mask():
    # Test set = every image that is not in the training set
    mask = np.ones((num_subjects, images_per_subject), dtype=bool)
    mask[:train_subjects, :train_images] = False

    return mask


def candidate_thresholds(scores):
    # Midpoints between consecutive sorted scores (+ one below / above all)
    s = np.unique(scores)
    return np.concatenate(([s[0] - 1.0], (s[:-1] + s[1:]) / 2, [s[-1] + 1.0]))


def tune_parameters(faces, train, train_labels, mean, eigenvectors, k_values):
    truth = ground_truth()
    mask = test_mask()

    best = dict(accuracy=-1.0)
    accuracy_per_k = []

    for k in k_values:

        M1 = compute_M1(faces, mean, eigenvectors[:, :k])[mask]
        M2, identities = compute_M2(faces, mean, eigenvectors[:, :k], train, train_labels)
        M2, identities = M2[mask], identities[mask]

        best_k = -1.0

        for face_threshold in candidate_thresholds(M1):

            # All known_threshold candidates at once (one row each)
            known_thresholds = candidate_thresholds(M2)
            decisions = decide(
                M1, M2, identities, face_threshold, known_thresholds[:, None]
            )
            accuracies = np.mean(decisions == truth[mask], axis=1)

            j = np.argmax(accuracies)
            best_k = max(best_k, accuracies[j])

            if accuracies[j] > best["accuracy"]:
                best = dict(
                    accuracy=accuracies[j],
                    num_eigenfaces=k,
                    face_threshold=face_threshold,
                    known_threshold=known_thresholds[j],
                )

        accuracy_per_k.append(best_k)

    return best, np.array(accuracy_per_k)


def plot_accuracy_vs_k(k_values, accuracy_per_k, best_k):
    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.plot(k_values, 100 * accuracy_per_k, color="#2a78d6", linewidth=2)

    best_acc = 100 * accuracy_per_k[list(k_values).index(best_k)]
    ax.plot(best_k, best_acc, "o", color="#2a78d6", markersize=8,
            markeredgecolor="white", markeredgewidth=2)
    ax.annotate(f"k = {best_k}: {best_acc:.1f}%", (best_k, best_acc),
                xytext=(10, 8), textcoords="offset points", color="#0b0b0b")

    ax.set_ylim(top=best_acc + 4)
    ax.set_xlabel("Number of eigenfaces k", color="#52514e")
    ax.set_ylabel("Test accuracy (%)", color="#52514e")
    ax.set_title("Test accuracy vs. k (thresholds tuned for each k)", color="#0b0b0b")
    ax.grid(color="#e6e5e1", linewidth=0.8)
    ax.tick_params(colors="#52514e")

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#c3c2b7")

    plt.tight_layout()
    return fig


def show_matrix(ax, M, title, **imshow_kwargs):
    im = ax.imshow(M, cmap="jet", aspect="auto", **imshow_kwargs)

    ax.set_title(title, fontsize=13)
    ax.set_xticks(range(images_per_subject))
    ax.set_xticklabels(range(1, images_per_subject + 1))
    ax.set_yticks(range(num_subjects))
    ax.set_yticklabels(range(1, num_subjects + 1), fontsize=7)
    ax.set_xlabel("Image index")
    ax.set_ylabel("Subject index")

    # Training block, and separators between known / unknown / non-face subjects
    ax.add_patch(Rectangle(
        (-0.5, -0.5), train_images, train_subjects,
        fill=False, edgecolor="white", linewidth=2, linestyle="--"
    ))
    for y in [known_subjects - 0.5, face_subjects - 0.5]:
        ax.axhline(y, color="white", linewidth=2)

    return im


if __name__ == "__main__":

    output_dir.mkdir(exist_ok=True)

    faces = load_dataset(data_path)
    train, train_labels = training_set(faces)

    mean, eigenvalues, eigenvectors = compute_pca(train)

    # Tuning
    k_values = range(1, 101)
    best, accuracy_per_k = tune_parameters(
        faces, train, train_labels, mean, eigenvectors, k_values
    )

    print("Best parameters found on the test set:")
    print(f"  num_eigenfaces  = {best['num_eigenfaces']}")
    print(f"  face_threshold  = {best['face_threshold']:.4f}")
    print(f"  known_threshold = {best['known_threshold']:.4f}")
    print(f"  test accuracy   = {100 * best['accuracy']:.2f}%\n")

    plot_accuracy_vs_k(k_values, accuracy_per_k, best["num_eigenfaces"])
    plt.savefig(output_dir / "q4d1.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Tuned method
    k = tuned_params["num_eigenfaces"]
    face_threshold = tuned_params["face_threshold"]
    known_threshold = tuned_params["known_threshold"]

    M1 = compute_M1(faces, mean, eigenvectors[:, :k])
    M2, identities = compute_M2(faces, mean, eigenvectors[:, :k], train, train_labels)
    M3 = decide(M1, M2, identities, face_threshold, known_threshold)

    truth = ground_truth()
    mask = test_mask()
    correct = M3 == truth

    groups = [
        ("known faces (s1-30, images 9-10)", mask[:known_subjects], correct[:known_subjects]),
        ("unknown faces (s31-37)", mask[known_subjects:face_subjects], correct[known_subjects:face_subjects]),
        ("non-faces (s38-40)", mask[face_subjects:], correct[face_subjects:]),
    ]

    print(f"Tuned method: k = {k}, face_threshold = {face_threshold}, "
          f"known_threshold = {known_threshold}")
    for label, m, c in groups:
        print(f"  {label:34s}: {c[m].sum():3d} / {m.sum():3d} correct")
    print(f"  Test accuracy: {correct[mask].sum()} / {mask.sum()} = "
          f"{100 * correct[mask].mean():.2f}%")
    print(f"  Accuracy on all 400 images (training + test): {100 * correct.mean():.2f}%")

    # M1, M2, M3
    fig, ax = plt.subplots(1, 3, figsize=(20, 13))

    im = show_matrix(ax[0], M1, f"M1: distance from face space\n"
                                f"(not a face if > {face_threshold:.3f})")
    fig.colorbar(im, ax=ax[0], fraction=0.05, pad=0.03)

    im = show_matrix(ax[1], M2, f"M2: distance to nearest training face\n"
                                f"(unknown person if > {known_threshold:.3f})")
    fig.colorbar(im, ax=ax[1], fraction=0.05, pad=0.03)

    im = show_matrix(ax[2], M3, "M3: decisions\n(-1 = not a face, 0 = unknown, "
                                "else subject index)", vmin=-1, vmax=known_subjects)
    cbar = fig.colorbar(im, ax=ax[2], fraction=0.05, pad=0.03)
    cbar.set_ticks([-1, 0, 5, 10, 15, 20, 25, 30])

    for s in range(num_subjects):
        for i in range(images_per_subject):

            ax[2].text(i, s, str(M3[s, i]), ha="center", va="center", fontsize=7,
                       path_effects=[pe.withStroke(linewidth=2, foreground="white")])

            # Outline wrong decisions
            if not correct[s, i]:
                ax[2].add_patch(Rectangle(
                    (i - 0.45, s - 0.45), 0.9, 0.9,
                    fill=False, edgecolor="black", linewidth=2
                ))

    fig.suptitle(
        f"Tuned method: k = {k} eigenfaces, test accuracy = "
        f"{100 * correct[mask].mean():.2f}%   (dashed box = training set, "
        f"black boxes in M3 = wrong decisions)",
        fontsize=14,
    )

    plt.tight_layout()
    plt.savefig(output_dir / "q4d2.png", dpi=150, bbox_inches="tight")
    plt.show()
