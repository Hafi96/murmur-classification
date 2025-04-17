#!/usr/bin/env python

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    accuracy_score, confusion_matrix, roc_curve, precision_recall_curve
)

from helper_code import load_patient_data, get_murmur, load_challenge_outputs, compare_strings

# File finding
def find_challenge_files(label_folder, output_folder):
    label_files, output_files = [], []
    for label_file in sorted(os.listdir(label_folder)):
        label_file_path = os.path.join(label_folder, label_file)
        if os.path.isfile(label_file_path) and label_file.lower().endswith('.txt'):
            root, _ = os.path.splitext(label_file)
            output_file_path = os.path.join(output_folder, root + '.csv')
            if os.path.isfile(output_file_path):
                label_files.append(label_file_path)
                output_files.append(output_file_path)
            else:
                print(f"⚠️ Warning: Missing output file for {label_file}")
    return label_files, output_files


def load_murmurs(label_files):
    valid_indices, labels = [], []
    for i, file in enumerate(label_files):
        data = load_patient_data(file)
        label = get_murmur(data)
        if label in ["Present", "Absent"]:
            labels.append([int(label == "Present"), int(label == "Absent")])
            valid_indices.append(i)
    return np.array(labels, dtype=int), valid_indices

# Output loading
def load_classifier_outputs(output_files, valid_indices):
    binary_outputs, scalar_outputs = [], []
    filtered_output_files = [output_files[i] for i in valid_indices]
    for file in filtered_output_files:
        _, patient_classes, _, patient_scalar_outputs = load_challenge_outputs(file)
        binary_output, scalar_output = [0, 0], [0.0, 0.0]
        for j, x in enumerate(["Present", "Absent"]):
            for k, y in enumerate(patient_classes):
                if compare_strings(x, y):
                    scalar_output[j] = patient_scalar_outputs[k]
                    binary_output[j] = int(patient_scalar_outputs[k] >= 0.5)
        binary_outputs.append(binary_output)
        scalar_outputs.append(scalar_output)
    return np.array(binary_outputs, dtype=int), np.array(scalar_outputs, dtype=np.float64)

# Metrics
def compute_auc(labels, outputs):
    try:
        auroc_p = roc_auc_score(labels[:, 0], outputs[:, 0])
        auprc_p = average_precision_score(labels[:, 0], outputs[:, 0])
        auroc_a = roc_auc_score(labels[:, 1], outputs[:, 1])
        auprc_a = average_precision_score(labels[:, 1], outputs[:, 1])
    except ValueError:
        auroc_p, auprc_p, auroc_a, auprc_a = 0.5, 0.5, 0.5, 0.5
    return [auroc_p, auroc_a], [auprc_p, auprc_a]

def compute_f_measure(labels, outputs):
    scores = [f1_score(labels[:, i], outputs[:, i]) for i in range(2)]
    return np.mean(scores), scores

def compute_accuracy(labels, outputs):
    scores = [accuracy_score(labels[:, i], outputs[:, i]) for i in range(2)]
    return np.mean(scores), scores

def compute_weighted_accuracy(labels, outputs):
    weights = np.array([[3, 1], [1, 2]])
    confusion = np.zeros((2, 2))
    for i in range(len(labels)):
        confusion[np.argmax(outputs[i]), np.argmax(labels[i])] += 1
    return np.trace(weights * confusion) / np.sum(weights * confusion)

# Visualizations
def generate_visualizations_multiclass(true_onehot, predicted_probs, class_names=["Present", "Absent"], output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)
    y_true = np.argmax(true_onehot, axis=1)
    y_pred = np.argmax(predicted_probs, axis=1)

    # ROC
    fpr, tpr, _ = roc_curve(true_onehot.ravel(), predicted_probs.ravel())
    plt.figure()
    plt.plot(fpr, tpr, label="Overall ROC")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Overall ROC Curve")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(output_dir, "overall_roc.png"))
    plt.close()

    # PR
    precision, recall, _ = precision_recall_curve(true_onehot.ravel(), predicted_probs.ravel())
    plt.figure()
    plt.plot(recall, precision, label="Overall PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Overall Precision-Recall Curve")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(output_dir, "overall_pr.png"))
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Murmur Detection')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(output_dir, "overall_confusion_matrix_multiclass.png"))
    plt.close()

# Evaluation runner
def evaluate_model(label_folder, output_folder):
    print("🔍 Evaluating model...")
    label_files, output_files = find_challenge_files(label_folder, output_folder)
    murmur_labels, valid_indices = load_murmurs(label_files)
    murmur_binary_outputs, murmur_scalar_outputs = load_classifier_outputs(output_files, valid_indices)

    threshold = 0.5
    murmur_binary_outputs = (murmur_scalar_outputs >= threshold).astype(int)

    generate_visualizations_multiclass(murmur_labels, murmur_scalar_outputs, ["Present", "Absent"])

    auroc, auprc = compute_auc(murmur_labels, murmur_scalar_outputs)
    f_measure, f_classes = compute_f_measure(murmur_labels, murmur_binary_outputs)
    accuracy, acc_classes = compute_accuracy(murmur_labels, murmur_binary_outputs)
    weighted_acc = compute_weighted_accuracy(murmur_labels, murmur_binary_outputs)

    return ["Present", "Absent"], auroc, auprc, f_measure, f_classes, accuracy, acc_classes, weighted_acc

# Save scores
def print_and_save_scores(filename, murmur_scores):
    classes, auroc, auprc, f_measure, f_measure_classes, accuracy, accuracy_classes, weighted_accuracy = murmur_scores
    output_string = f"""
# Murmur scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy
{np.mean(auroc):.3f},{np.mean(auprc):.3f},{f_measure:.3f},{accuracy:.3f},{weighted_accuracy:.3f}

# Murmur scores (per class)
Classes,Present,Absent
AUROC,{auroc[0]:.3f},{auroc[1]:.3f}
AUPRC,{auprc[0]:.3f},{auprc[1]:.3f}
F-measure,{f_measure_classes[0]:.3f},{f_measure_classes[1]:.3f}
Accuracy,{accuracy_classes[0]:.3f},{accuracy_classes[1]:.3f}
"""
    print(output_string)
    with open(filename, 'w') as f:
        f.write(output_string.strip())
    print(f"✅ Scores saved to {filename}")

# Entry point
if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python evaluate_model_murmur.py <label_folder> <output_folder> <scores.csv>")
        sys.exit(1)

    murmur_scores = evaluate_model(sys.argv[1], sys.argv[2])
    print_and_save_scores(sys.argv[3], murmur_scores)
    print(" Evaluation complete. Check scores.csv and plots/ for output visuals.")
