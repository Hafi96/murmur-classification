DATALOADER_CFG = {
    'batch_size': 128,
    'num_workers': 4,
    'pin_memory': True,
}

DATASET_CFG = {
    'num_murmur_classes': 2,
    # 'num_outcome_classes': 2, 
    'murmur_classes': ['Present','Absent'],
    # 'outcome_classes': ['Abnormal', 'Normal']
}

PREPROCESSING_CFG = {
    # Cropped segment length
    'length': 3,
    # Default sampling frequnecy: 4000
    'frequency': 2000,
    # Crop the head of tails of the raw recording (in seconds)
    'head_crop': 1.0,
    # Spectrograms Scales:
    'scales': [1.0, 0.5, 0.25],
    # If true, Normalize the augmented recording
    'normalize': True,
}


SPEC_AUGMENTATION_CFG = {
    'freq_mask_prob': 0.5,
    'freq_mask_param': 20,
    'time_mask_prob': 0.5,
    'time_mask_param': 40,
}

MODEL_CFG = {
    'stagewise_layers': [2, 2, 2, 2],
    'scalewise_inplanes': [32, 16, 16]
}

# Adam
OPTIMIZER_CFG={
    'lr': 1e-3,
    'weight_decay': 0.,
    'amsgrad': False,
}

TRAINING_CFG = {
    'epochs': 100,
    'print_freq': 50,
    'label_smoothing': 0.1,
}