#!/usr/bin/env python

import os
import sys
import numpy as np
from helper_code import load_patient_data, get_murmur, load_challenge_outputs, compare_strings
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

# ✅ Function to find label and output files
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

# ✅ Function to load murmur labels
def load_murmurs(label_files):
    valid_indices, labels = [], []
    for i, file in enumerate(label_files):
        data = load_patient_data(file)
        label = get_murmur(data)
        if label in ["Present", "Absent"]:  # Ignore "Unknown"
            labels.append([int(label == "Present"), int(label == "Absent")])
            valid_indices.append(i)
    return np.array(labels, dtype=int), valid_indices

# ✅ Function to load classifier outputs
def load_classifier_outputs(output_files, valid_indices):
    binary_outputs, scalar_outputs = [], []
    filtered_output_files = [output_files[i] for i in valid_indices]
    for file in filtered_output_files:
        _, patient_classes, _, patient_scalar_outputs = load_challenge_outputs(file)
        binary_output, scalar_output = [0, 0], [0.0, 0.0]  # Default
        for j, x in enumerate(["Present", "Absent"]):
            for k, y in enumerate(patient_classes):
                if compare_strings(x, y):
                    scalar_output[j] = patient_scalar_outputs[k]
                    binary_output[j] = int(patient_scalar_outputs[k] >= 0.5)  # Default threshold
        binary_outputs.append(binary_output)
        scalar_outputs.append(scalar_output)
    return np.array(binary_outputs, dtype=int), np.array(scalar_outputs, dtype=np.float64)

# ✅ Compute the best threshold using F1-score

# ✅ Compute evaluation metrics
def compute_auc(labels, outputs):
    try:
        auroc_present = roc_auc_score(labels[:, 0], outputs[:, 0])
        auprc_present = average_precision_score(labels[:, 0], outputs[:, 0])
        auroc_absent = roc_auc_score(labels[:, 1], outputs[:, 1])
        auprc_absent = average_precision_score(labels[:, 1], outputs[:, 1])
    except ValueError:
        auroc_present, auprc_present, auroc_absent, auprc_absent = 0.5, 0.5, 0.5, 0.5
    return (auroc_present, auprc_present, auroc_absent, auprc_absent)

def compute_f_measure(labels, outputs):
    f1_present = f1_score(labels[:, 0], outputs[:, 0])
    f1_absent = f1_score(labels[:, 1], outputs[:, 1])
    return np.mean([f1_present, f1_absent]), [f1_present, f1_absent]

def compute_accuracy(labels, outputs):
    accuracy_present = accuracy_score(labels[:, 0], outputs[:, 0])
    accuracy_absent = accuracy_score(labels[:, 1], outputs[:, 1])
    return np.mean([accuracy_present, accuracy_absent]), [accuracy_present, accuracy_absent]

def compute_weighted_accuracy(labels, outputs):
    weights = np.array([[5, 1], [5, 1]])
    confusion = np.zeros((2, 2))
    for i in range(len(labels)):
        confusion[np.argmax(outputs[i]), np.argmax(labels[i])] += 1
    weighted_acc = np.trace(weights * confusion) / np.sum(weights * confusion)
    return weighted_acc

# ✅ Main evaluation function
def evaluate_model(label_folder, output_folder):
    print("🔍 Evaluating model...")

    # Load label & output files
    label_files, output_files = find_challenge_files(label_folder, output_folder)
    murmur_labels, valid_indices = load_murmurs(label_files)
    murmur_binary_outputs, murmur_scalar_outputs = load_classifier_outputs(output_files, valid_indices)

    # Find best threshold
    threshold = 0.5

    # Apply threshold
    murmur_binary_outputs = (murmur_scalar_outputs >= threshold).astype(int)

    # Compute evaluation metrics
    auroc_present, auprc_present, auroc_absent, auprc_absent = compute_auc(murmur_labels, murmur_scalar_outputs)
    murmur_f_measure, murmur_f_measure_classes = compute_f_measure(murmur_labels, murmur_binary_outputs)
    murmur_accuracy, murmur_accuracy_classes = compute_accuracy(murmur_labels, murmur_binary_outputs)
    murmur_weighted_accuracy = compute_weighted_accuracy(murmur_labels, murmur_binary_outputs)

    return ["Present", "Absent"], [auroc_present, auroc_absent], [auprc_present, auprc_absent], \
           murmur_f_measure, murmur_f_measure_classes, murmur_accuracy, murmur_accuracy_classes, murmur_weighted_accuracy

# ✅ Print & Save scores
def print_and_save_scores(filename, murmur_scores):
    classes, auroc, auprc, f_measure, f_measure_classes, accuracy, accuracy_classes, weighted_accuracy = murmur_scores
    total_auroc= np.mean([auroc[0],auroc[1]])
    total_auprc = np.mean([auprc[0], auprc[1]])
    output_string = f"""
#Murmur scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy
{total_auroc:.3f},{total_auprc:.3f},{f_measure:.3f},{accuracy:.3f},{weighted_accuracy:.3f}

#Murmur scores (per class)
Classes,Present,Absent
AUROC,{auroc[0]:.3f},{auroc[1]:.3f}
AUPRC,{auprc[0]:.3f},{auprc[1]:.3f}
F-measure,{f_measure_classes[0]:.3f},{f_measure_classes[1]:.3f}
Accuracy,{accuracy_classes[0]:.3f},{accuracy_classes[1]:.3f}
"""

    # ✅ Print results to console
    print(output_string)

    # ✅ Save to file
    with open(filename, 'w') as f:
        f.write(output_string.strip())
    print(f"✅ Scores saved to {filename}")

# ✅ Run the evaluation script
if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python evaluate_model.py <label_folder> <output_folder> <scores.csv>")
        sys.exit(1)

    murmur_scores = evaluate_model(sys.argv[1], sys.argv[2])
    print_and_save_scores(sys.argv[3], murmur_scores)

    print("✅ Model Evaluation Completed. Check scores.csv for detailed results.")
