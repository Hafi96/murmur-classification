import os
import math
import time
import random
import numpy as np
from scipy.io import wavfile
from scipy import signal
from librosa import effects
import torch
import torch.nn as nn
from torchaudio import transforms
from torch.utils.data import Dataset
from utils import load_recordings_with_labels, load_patient_features
from helper_code import load_patient_data


def normalize_recording(recordings, dim=1, eps=1e-8):
    mean, std = recordings.mean(dim=dim, keepdims=True), recordings.std(dim=dim, keepdims=True)
    recordings = (recordings - mean) / (std + eps)
    return recordings

def build_spectrogram_transform(scale=1.0):
    nfft = int((224 * scale - 1) * 2)
    win_length = int(200 * scale)
    hop_length = int(27 / scale)
    return transforms.Spectrogram(n_fft=nfft, win_length=win_length, hop_length=hop_length, normalized=True, power=1.)

class PCGDataset(Dataset):
    def __init__(self, 
                 data_folder, 
                 classes=['Present', 'Absent'],  # Removed 'Unknown'
                 preprocessor=None,
                 target='murmur'
                ):
        self.data_folder = data_folder
        self.classes = classes
        self.num_classes = len(self.classes)
        self.patient_files, self.recordings, self.murmurs= load_recordings_with_labels(self.data_folder, self.classes)
        self.preprocessor = preprocessor
        self.target = target
        
    def __len__(self):
        return len(self.recordings)
    
    def __getitem__(self, idx):
        freq, recording = wavfile.read(self.recordings[idx])
        patient_data = load_patient_data(self.patient_files[idx])
        patient_features = load_patient_features(patient_data)
        spec, ratio = self.preprocessor(recording, freq)
        murmur = self.murmurs[idx]
       
        # outcome = self.outcomes[idx]
        if self.target == 'murmur':
            return spec, patient_features, murmur
        # elif self.target == 'outcome':
        #     return spec, patient_features, outcome
        else:
            raise NotImplementedError('Unknown target')

    
class Preprocessor:
    def __init__(self, frequency, normalize=True, 
                 length=3, head_crop=0,
                 scales=[1.0, 0.5, 0.25], 
                 mode='train'):
        self.frequency = frequency
        self.normalize = normalize
        self.crop_size = int(length * self.frequency)
        self.head_crop = int(head_crop * self.frequency)
        self.transform = [build_spectrogram_transform(scale=scale) for scale in scales]
        self.mode = mode
        
    def random_crop(self, recording):
        maxmium_end_idx = len(recording) - self.crop_size + 1
        start_idx = np.random.randint(maxmium_end_idx)
        recording = recording[start_idx: start_idx + self.crop_size]
        return recording
    
    def measure_quality(self, spectrogram, low=20, high=200):
        split_start = round(2 * 224 / self.frequency * low)
        split_end = round(2 * 224 / self.frequency * high)
        quality = spectrogram[split_start:split_end, :].sum() / spectrogram.sum()
        return quality

    def search_crop(self, recording, interval=0.5):
        if self.mode == 'train':
            return self.random_crop(recording)
        elif self.mode == 'valid':
            return self.random_crop(recording)
        else:
            maxmium_end_idx = len(recording) - self.crop_size + 1
            interval = int(interval * self.frequency)
            all_start_indicies = np.arange(0, maxmium_end_idx, interval)
            recordings = np.stack([recording[si: si+self.crop_size] for si in all_start_indicies])
            return recordings
        
    def __call__(self, recording, freq, interval=1.0):
        # adjust sampling frequency
        recording = signal.resample(recording, int(len(recording) * self.frequency / freq))
        # From Int to Float
        recording = recording / 2**15
        # Crop head of tails of recording
        if self.head_crop > 0 and len(recording) > 5 * self.frequency:
            recording = recording[self.head_crop: -self.head_crop]
        
        if len(recording) < self.crop_size:
            recording = np.pad(recording, self.crop_size - len(recording))
        
        if self.mode != 'test':
            recording = self.search_crop(recording)
            recording = recording[np.newaxis, :]
        else:
            recording = self.search_crop(recording, interval)
            
        # Transfer from ndarray to tensor
        recording = torch.from_numpy(recording).float()
        if self.normalize:
            recording = normalize_recording(recording, dim=1)
        if self.transform:
            multi_scale_specs = [t(recording) for t in self.transform]
            if self.mode == 'test':
                for i in range(len(multi_scale_specs)):
                    multi_scale_specs[i] = multi_scale_specs[i].unsqueeze(1)
                ratio = np.array([self.measure_quality(multi_scale_specs[0][i][0]) for i in range(multi_scale_specs[0].shape[0])])
            else:
                ratio = self.measure_quality(multi_scale_specs[0][0])
                
            multi_scale_specs = [s.abs().clamp(min=1e-8).log() for s in multi_scale_specs]

        return multi_scale_specs, ratio
