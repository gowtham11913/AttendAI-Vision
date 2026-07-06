"""
evaluate_model.py
------------------
Proper accuracy reporting for the SVM face recognition model - goes
beyond the single accuracy number printed by train_model.py.

What train_model.py's accuracy means today: with few images per student,
there often isn't enough data for a held-out test split, so that number
can be measured on the training data itself - which is optimistic and
doesn't tell you how the model behaves on a NEW photo of the same person.
This script does a proper stratified train/test split (always, with a
sensible adjustment for small classes) and reports:

    - Overall accuracy on the held-out test set
    - Per-student precision / recall / F1 (sklearn classification_report)
    - A confusion matrix (printed as text, and saved as a PNG if
      matplotlib is available)
    - Which specific students are most often confused with each other

Run:
    python evaluate_model.py
"""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings", "students")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")


def load_dataset():
    npy_files = sorted(f for f in os.listdir(EMBEDDINGS_DIR) if f.endswith(".npy"))
    if not npy_files:
        print("No embeddings found in embeddings/students/. Run generate_embeddings.py first.")
        sys.exit(1)

    X_list, y_list = [], []
    for filename in npy_files:
        roll_no = os.path.splitext(filename)[0]
        arr = np.load(os.path.join(EMBEDDINGS_DIR, filename))
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        X_list.append(arr)
        y_list.extend([roll_no] * arr.shape[0])

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list)
    return X, y


def main():
    print("Loading embeddings...")
    X, y = load_dataset()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = label_encoder.classes_

    counts = np.bincount(y_encoded)
    min_count = counts.min()

    if min_count < 2:
        print("ERROR: At least 2 images are required per student to evaluate "
              "(some students have only 1). Re-register with more images, "
              "or run generate_embeddings.py if registration already captured "
              "enough images but embeddings are stale.")
        sys.exit(1)

    # With very few images per student, a 50/50 split keeps at least 1
    # sample per class in both sets while still giving a genuine held-out
    # test. With more data, a standard 80/20 split is used.
    test_size = 0.5 if min_count < 5 else 0.2

    print(f"Total samples: {len(y)}  |  Students: {len(class_names)}  |  "
          f"Min images/student: {min_count}  |  Test split: {test_size:.0%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, stratify=y_encoded, random_state=42
    )

    print("\nTraining SVM on the training split only (held-out test data "
          "was NOT seen during this fit)...")
    model = SVC(kernel="linear", probability=True, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print(f"HELD-OUT TEST ACCURACY: {accuracy * 100:.2f}%")
    print(f"({len(y_test)} test samples, never seen during training)")
    print("=" * 60)

    print("\nPer-student precision / recall / F1:\n")
    present_labels = sorted(set(y_test) | set(y_pred))
    report = classification_report(
        y_test, y_pred,
        labels=present_labels,
        target_names=[class_names[i] for i in present_labels],
        zero_division=0,
    )
    print(report)

    cm = confusion_matrix(y_test, y_pred, labels=present_labels)
    cm_names = [class_names[i] for i in present_labels]

    print("Confusion matrix (rows = actual, columns = predicted):\n")
    header = "        " + "".join(f"{n:>10}" for n in cm_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "".join(f"{v:>10}" for v in row)
        print(f"{cm_names[i]:>8}{row_str}")

    # Flag specific confusions (off-diagonal entries > 0) so you know
    # exactly which students get mixed up with each other.
    print("\nMisclassifications found:")
    found_any = False
    for i, actual in enumerate(cm_names):
        for j, predicted in enumerate(cm_names):
            if i != j and cm[i][j] > 0:
                found_any = True
                print(f"  {actual} was predicted as {predicted}: {cm[i][j]} time(s)")
    if not found_any:
        print("  None - every test sample was classified correctly.")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "recognition_accuracy_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Held-out test accuracy: {accuracy * 100:.2f}%\n")
        f.write(f"Test samples: {len(y_test)}\n\n")
        f.write(report)
        f.write("\nConfusion matrix (rows=actual, cols=predicted):\n")
        f.write(header + "\n")
        for i, row in enumerate(cm):
            row_str = "".join(f"{v:>10}" for v in row)
            f.write(f"{cm_names[i]:>8}{row_str}\n")
    print(f"\nText report saved to: {report_path}")

    # Optional visual confusion matrix, only if matplotlib is installed.
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(max(6, len(cm_names) * 0.8), max(5, len(cm_names) * 0.8)))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(cm_names)))
        ax.set_yticks(range(len(cm_names)))
        ax.set_xticklabels(cm_names, rotation=45, ha="right")
        ax.set_yticklabels(cm_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Face Recognition Confusion Matrix")
        for i in range(len(cm_names)):
            for j in range(len(cm_names)):
                ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                         color="white" if cm[i][j] > cm.max() / 2 else "black")
        fig.colorbar(im)
        fig.tight_layout()
        png_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
        fig.savefig(png_path, dpi=150)
        print(f"Confusion matrix image saved to: {png_path}")
    except ImportError:
        print("\n(matplotlib not installed - skipped saving confusion_matrix.png. "
              "Run: pip install matplotlib --break-system-packages   to enable it.)")

    print("\nNOTE: This evaluation trains a temporary model on a split for "
          "reporting purposes only. It does NOT overwrite your deployed "
          "models/svm_model.pkl - that still comes from train_model.py, "
          "which refits on ALL data for best real-world performance.")


if __name__ == "__main__":
    main()
