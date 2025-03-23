#!/usr/bin/env python

# Do *not* edit this script.
# These are helper functions that you can use with your code.

import os, numpy as np, scipy as sp, scipy.io, scipy.io.wavfile

# Check if a variable is a number or represents a number.
def is_number(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False

# Check if a variable is an integer or represents an integer.
def is_integer(x):
    if is_number(x):
        return float(x).is_integer()
    else:
        return False

# Check if a variable is a finite number or represents a finite number.
def is_finite_number(x):
    if is_number(x):
        return np.isfinite(float(x))
    else:
        return False

# Compare normalized strings.
def compare_strings(x, y):
    try:
        return str(x).strip().casefold() == str(y).strip().casefold()
    except AttributeError:  # For Python 2.x compatibility
        return str(x).strip().lower() == str(y).strip().lower()

# Find patient data files.
def find_patient_files(data_folder):
    filenames = sorted([os.path.join(data_folder, f) for f in os.listdir(data_folder)
                        if f.endswith('.txt') and not f.startswith('.')])
    
    # Sort numerically if filenames are integers.
    roots = [os.path.splitext(os.path.basename(f))[0] for f in filenames]
    if all(is_integer(root) for root in roots):
        filenames = sorted(filenames, key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))

    return filenames
# Load Challenge outputs.
# Load Challenge outputs.
def load_challenge_outputs(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    if len(lines) < 4:
        raise ValueError(f"Invalid output file format in {filename}")

    # Detect delimiter (use tab if found, otherwise fallback to comma)
    delimiter = '\t' if '\t' in lines[1] else ','

    # Extract patient ID
    patient_id = lines[0].replace('#', '').strip()

    # Extract classes
    classes = lines[1].strip().split(delimiter)

    # Extract labels (convert to integers)
    labels = list(map(int, lines[2].strip().split(delimiter)))

    # Extract probabilities (convert to 0 or 1)
    probabilities = list(map(lambda x: int(round(float(x))), lines[3].strip().split(delimiter)))

    return patient_id, classes, labels, probabilities

# Load patient data as a string.
def load_patient_data(filename):
    with open(filename, 'r') as f:
        data = f.read()
    return data

# Load a WAV file.
def load_wav_file(filename):
    frequency, recording = sp.io.wavfile.read(filename)
    return recording, frequency

# Load recordings.
def load_recordings(data_folder, data, get_frequencies=False):
    num_locations = get_num_locations(data)
    recording_information = data.split('\n')[1:num_locations + 1]

    recordings, frequencies = [], []
    for i in range(num_locations):
        entries = recording_information[i].split(' ')
        filename = os.path.join(data_folder, entries[2])
        recording, frequency = load_wav_file(filename)
        recordings.append(recording)
        frequencies.append(frequency)

    return (recordings, frequencies) if get_frequencies else recordings

# Get patient ID from patient data.
def get_patient_id(data):
    return data.split('\n')[0].split(' ')[0].strip()

# Get number of recording locations from patient data.
def get_num_locations(data):
    return int(data.split('\n')[0].split(' ')[1])

# Get frequency from patient data.
def get_frequency(data):
    return float(data.split('\n')[0].split(' ')[2])

# Get recording locations from patient data.
def get_locations(data):
    return [line.split(' ')[0] for line in data.split('\n')[1:get_num_locations(data) + 1]]

# Get patient attributes from patient data.
def get_age(data):
    return next((line.split(': ')[1].strip() for line in data.split('\n') if line.startswith('#Age:')), None)

def get_sex(data):
    return next((line.split(': ')[1].strip() for line in data.split('\n') if line.startswith('#Sex:')), None)

def get_height(data):
    return float(next((line.split(': ')[1].strip() for line in data.split('\n') if line.startswith('#Height:')), 'nan'))

def get_weight(data):
    return float(next((line.split(': ')[1].strip() for line in data.split('\n') if line.startswith('#Weight:')), 'nan'))

def get_pregnancy_status(data):
    return bool(sanitize_binary_value(next((line.split(': ')[1].strip() for line in data.split('\n') if line.startswith('#Pregnancy status:')), 0)))

# Get murmur status from patient data.
def get_murmur(data):
    murmur = next((line.split(': ')[1].strip() for line in data.split('\n') if line.startswith('#Murmur:')), None)
    if murmur is None:
        raise ValueError('No murmur available. Is your code trying to load labels from the hidden data?')
    return murmur

# Sanitize binary values.
def sanitize_binary_value(x):
    x = str(x).replace('"', '').replace("'", "").strip()
    return 1 if is_finite_number(x) and float(x) == 1 else 0

# Sanitize scalar values.
def sanitize_scalar_value(x):
    x = str(x).replace('"', '').replace("'", "").strip()
    return float(x) if is_finite_number(x) else 0.0

# Save Challenge outputs.
# Save Challenge outputs (Only "Present" and "Absent" with Correct Formatting)
def save_challenge_outputs(filename, patient_id, classes, labels, probabilities):
    patient_string = f'#{patient_id}'

    # Expected classes in the correct order
    expected_classes = ['Present', 'Absent']

    # Map classes to values
    label_dict = dict(zip(classes, labels))
    probability_dict = dict(zip(classes, probabilities))

    # Assign values based on expected order, default to 0
    ordered_labels = [int(label_dict.get(cls, 0)) for cls in expected_classes]
    ordered_probabilities = [f"{probability_dict.get(cls, 0.0):.1f}" for cls in expected_classes]

    label_string = ','.join(map(str, ordered_labels))
    probabilities_string = ','.join(ordered_probabilities)

    output_lines = [
        patient_string,
        ','.join(expected_classes),
        label_string,
        probabilities_string
    ]

    with open(filename, 'w') as f:
        for line in output_lines:
            f.write(line + '\n')
