import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
from pathlib import Path
import numpy as np
from config import IMG_SIZE, MEAN, STD, NUM_WORKERS, PIN_MEMORY, AUG_POLICIES

# Change augmentations method if needed
def get_transforms(policy:str, split:str):
    if split=='train':
        if policy=='light':
            return transforms.Compose([
                transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(MEAN,STD),
            ])
        else:
            return transforms.Compose([
                transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(25),
                transforms.ColorJitter(0.3,0.3,0.2,0.02),
                transforms.RandomApply([transforms.GaussianBlur((3,3))],p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(MEAN,STD),
            ])
    else:
        return transforms.Compose([
            transforms.Resize(int(IMG_SIZE*1.15)),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(MEAN,STD),
        ])

# Make sure your dataset folder already split in three folders train,val,test
class LeafImageDataset:
    def __init__(self,data_root:Path,dataset_key:str,batch_size:int):
        self.dataset_root = Path(data_root)
        self.policy = AUG_POLICIES.get(dataset_key,'strong')
        self.batch_size = batch_size
        self.train_dir = self.dataset_root/'train'
        self.val_dir = self.dataset_root/'val'
        self.test_dir = self.dataset_root/'test'
        self._build()

    def _build(self):
        train_tf = get_transforms(self.policy,'train')
        val_tf = get_transforms(self.policy,'val')
        test_tf = get_transforms(self.policy,'test')
        self.train_set = ImageFolder(self.train_dir, transform=train_tf)
        self.val_set = ImageFolder(self.val_dir, transform=val_tf)
        self.test_set = ImageFolder(self.test_dir, transform=test_tf)

        counts = np.bincount([y for _,y in self.train_set.samples])
        class_freq = counts/counts.sum()
        class_weights = 1.0/(class_freq+1e-8)
        sample_weights = [class_weights[y] for _,y in self.train_set.samples]
        sampler = WeightedRandomSampler(sample_weights,len(sample_weights),replacement=True)

        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, sampler=sampler,
                                       num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY)
        self.val_loader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                                     num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY)
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False,
                                      num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY)

        self.num_classes = len(self.train_set.classes)
        self.class_to_idx = self.train_set.class_to_idx
        self.idx_to_class = {v:k for k,v in self.class_to_idx.items()}

    def get_loaders(self):
        return self.train_loader,self.val_loader,self.test_loader
    def get_num_classes(self):
        return self.num_classes
    def get_class_names(self):
        return [self.idx_to_class[i] for i in range(self.num_classes)]
