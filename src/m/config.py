import os
from pathlib import Path
import torch

# Change the folder locations before you run the code
ROOT = Path('/data/URA/Model Performance Classification/dinov3_ConvNeXt')
DATA_ROOT = Path('/data/URA/Master Dataset/Preprocessed')

CHECKPOINT_DIR = ROOT / 'output' / 'checkpoints'
RESULTS_DIR = ROOT / 'output' / 'results'
CONFUSION_DIR = ROOT / 'output' / 'confusion_matrices'

for d in [CHECKPOINT_DIR, RESULTS_DIR, CONFUSION_DIR]:
    os.makedirs(d, exist_ok=True)

# Give the pretrained weight location that you wanna use
PRETRAINED_PATH = '/data/URA/Pretrained Weights/dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Add dataset name and its location that you wanna train into this dictionary
DATASETS = {
    'apple': DATA_ROOT / 'Apple Tree Leaf Disease Dataset',
    # 'cassava': DATA_ROOT / 'Cassava Leaf Disease Dataset',
    # 'corn': DATA_ROOT / 'Corn or Maize Leaf Disease Dataset',
    'potato': DATA_ROOT / 'Potato Leaf Disease Dataset',
    # 'rice': DATA_ROOT / 'Rice Leaf Disease Dataset',
    'plant_village': DATA_ROOT / 'Plant Village',
    'PDD_Pretrained': DATA_ROOT / 'PDD_Pretrained',
    'crop_pest': DATA_ROOT / 'Crop Pest and Disease Detection',
    'agricultural_pests': DATA_ROOT / 'Agricultural Pests',
}

# After that choose the augmentation policy. Strong if your dataset needs strong augmentation, and light if light agumentation.
AUG_POLICIES = {
    'apple': 'strong',
    'cassava': 'strong',
    'corn': 'light',
    'potato': 'strong',
    'rice': 'light',
    'plant_village': 'strong',
    'PDD_Pretrained': 'strong',
    'crop_pest': 'strong',
    'agricultural_pests': 'strong',
}

# List the (batch_size, num_epochs) that you'd like to experiment
EXPERIMENT_CONFIGS = [
    (16,5),(16,10),(32,10),(32,20),(48,20),(48,30),
    (64,30),(64,40),(80,40),(80,60),(96,60),(112,80),(128,90)
]

# Change the essential configs if you need
IMG_SIZE = 384
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

LR = 1e-4
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 6
PIN_MEMORY = True
SEED = 42
SHUFFLE_TRAIN = False
TOPK = (1,3,5)
