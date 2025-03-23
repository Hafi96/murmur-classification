#!/usr/bin/env python


################################################################################
#
# Import libraries and functions. You can change or remove them.
#
################################################################################

import os
from helper_code import *
import numpy as np
import torch
import torch.nn.functional as F

from config import *
from data import Preprocessor, PCGDataset
from torch.utils.data import DataLoader
from HMSSNet import Hierachical_MS_Net
from utils import AverageMeter, calc_accuracy, load_patient_features
from loss import LabelSmoothingCrossEntropy


# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    verbose = verbose >= 1
    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Build Datasets and Loaders
    if verbose: 
        print('Loading datasets...')
    train_preprocessor = Preprocessor(**PREPROCESSING_CFG, 
                                      mode = 'train')
    
    train_dataset = PCGDataset(data_folder, 
                               preprocessor = train_preprocessor, 
                               classes = DATASET_CFG['murmur_classes'],
                               target = 'murmur')
    train_loader = DataLoader(train_dataset, 
                              shuffle=True,
                              drop_last=True, 
                              **DATALOADER_CFG)
    
    if verbose:
        print('Building up Torch CNN and optimizer...')
    murmur_classifier = Hierachical_MS_Net(num_classes=DATASET_CFG['num_murmur_classes'], **MODEL_CFG).to(device)
    optimizer = torch.optim.AdamW(murmur_classifier.parameters(), **OPTIMIZER_CFG)
    criterion = LabelSmoothingCrossEntropy(TRAINING_CFG['label_smoothing'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, min_lr=1e-7, verbose=verbose)

    # Stage 1: Train the classifier for Murmur classification
    if verbose:
        print('Training model for murmur classification...')
    for epoch in range(TRAINING_CFG['epochs']):
        if verbose:
            print(f'Epoch {epoch} starts...')
        train_epoch(train_loader, murmur_classifier, optimizer, criterion, scheduler, device, TRAINING_CFG['print_freq'])
        if verbose:
            print('\n')
        save_challenge_model(model_folder, murmur_classifier, file_name='murmur_classifier')
        
    
        
    if verbose:
        print('Done.')
            
def train_epoch(dataloader, model, optimizer, criterion, scheduler=None, device='cuda', print_freq=10, verbose=False):
    model.train()
    acc_meter, loss_meter = AverageMeter(), AverageMeter()

    for i, (multi_scale_specs, patient_features, targets) in enumerate(dataloader):
        multi_scale_specs = [s.to(device) for s in multi_scale_specs]
        patient_features = patient_features.to(device)
        targets = targets.to(device)
        batch_size = targets.size(0)
        preds = model(multi_scale_specs, patient_features)
        targets = targets.to(torch.int64)  # Ensure targets are integers

        batch_loss = criterion(preds, targets)
        batch_acc = calc_accuracy(preds, targets)
        
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        
        loss_meter.update(batch_loss.item(), batch_size)
        acc_meter.update(batch_acc.item(), batch_size)
            
        if verbose and i != 0 and i % print_freq == 0:
            print(f'Training Iteration: {i} '\
                  f'Loss: {loss_meter.avg:.6f} \t'\
                  f'Accuracy: {acc_meter.avg*100:.4f}')
            
    print(f'Training Loss: {loss_meter.avg:.6f} \t'\
          f'Accuracy: {acc_meter.avg*100:.4f}')
            
    if scheduler:
        scheduler.step(loss_meter.avg)


def calc_pred_locations(preds, window_size=3, interval=0.5, freq=2000):
    interval = int(interval * freq)
    window_size = int(window_size * freq)
    recording_length = window_size + (preds.shape[0] - 1) * interval
    unknown_probs = []
    
    location_preds = np.zeros((recording_length, preds.shape[1]))
    for i in range(len(preds)):
        location_preds[i*interval: i*interval + window_size, :] += preds[i]
    location_preds = np.argmax(location_preds, -1)
    return location_preds


@torch.no_grad()
def recording_murmur_diagnose(multi_scale_specs, murmur_classifier, murmur_classes, interval):
    murmur_logits = murmur_classifier(multi_scale_specs)
    murmur_probs = F.softmax(murmur_logits, -1).cpu().numpy()
    location_preds = calc_pred_locations(murmur_probs, 
                                        window_size=PREPROCESSING_CFG['length'],
                                        interval=interval,
                                        freq=PREPROCESSING_CFG['frequency'])
    class_duration = np.bincount(location_preds, minlength=len(murmur_classes)) / PREPROCESSING_CFG['frequency']
    
    if class_duration[1] / sum(class_duration) > 0.8:
        pred = 1
    else:
        if class_duration[0] >= 3:
            pred = 0
        else:
            pred = 2
    return pred
    

@torch.no_grad()
def run_challenge_model(model, data, recordings, verbose):
    (device, preprocessor, murmur_classifier, murmur_classes) = model  
    interval = 1.0
    recording_murmur_counts = np.zeros(len(murmur_classes), dtype=np.int_)

    patient_features = torch.from_numpy(load_patient_features(data)).unsqueeze(0).to(device)
    recording_murmur_preds = np.zeros(len(recordings), dtype=np.int_)

    for i in range(len(recordings)):
        multi_scale_specs, qualities = preprocessor(recordings[i], 4000, interval=interval)
        multi_scale_specs = [s.to(device) for s in multi_scale_specs]
        recording_murmur_preds[i] = recording_murmur_diagnose(multi_scale_specs, murmur_classifier, murmur_classes, interval)

    recording_murmur_counts = np.bincount(recording_murmur_preds, minlength=len(murmur_classes))

    # Assign murmur labels based on the most frequent prediction
    murmur_labels = np.zeros(len(murmur_classes), dtype=np.int_)
    murmur_labels[np.argmax(recording_murmur_counts)] = 1
    classes=murmur_classes
    probabilities = recording_murmur_counts / np.sum(recording_murmur_counts) if np.sum(recording_murmur_counts) > 0 else np.zeros(len(murmur_classes))

    return classes, murmur_labels, probabilities
    


# Save the trained model.
def save_challenge_model(model_folder, model, file_name='murmur_classifier'):
    torch.save({
        'model_state_dict': model.state_dict(),
    }, os.path.join(model_folder, f'{file_name}.pth'))
    


def load_challenge_model(model_folder, verbose):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    preprocessor = Preprocessor(mode='test', **PREPROCESSING_CFG)
    
    murmur_checkpoint = torch.load(os.path.join(model_folder, 'murmur_classifier.pth'), map_location=device)
    murmur_classifier = Hierachical_MS_Net(num_classes=DATASET_CFG['num_murmur_classes'], **MODEL_CFG).to(device)
    murmur_classifier.load_state_dict(murmur_checkpoint['model_state_dict'])
    murmur_classifier.eval()
    

    
    murmur_classes = DATASET_CFG['murmur_classes']
   
    
    return (device, preprocessor, murmur_classifier, murmur_classes)
