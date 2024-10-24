import pandas as pd
import torch
import torchvision
from enum import Enum 
from PIL import Image

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Definition of Expressions
class Expression(Enum):
    NEUTRAL = "0"
    HAPPY = "1"
    SAD = "2"
    SURPRISE = "3"
    FEAR = "4"
    ANGER = "5"
    DISGUST = "6"
    CONTEMPT = "7"
    NONE = "8"
    UNCERTAIN = "9"
    NONFACE = "10"

class AffectNetDataset(Dataset):
    def __init__(self, image_paths, labels, transform):
        self.labels = labels
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        image = Image.open(f"dataset/images/{self.image_paths[idx]}").convert("RGB")
        return self.transform(image), self.labels[idx]

class AffectNetDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.train_files = pd.read_csv("dataset/filelists/training.csv") # shuffle if needed: .sample(frac=1, random_state=27)
        self.val_files = pd.read_csv("dataset/filelists/validation.csv", header=None) # might need to figure out later
        header = self.train_files.columns
        self.val_files.columns = header

        # Transform function for Image data
        self.transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Resize images to 224x224 for example
                transforms.ToTensor(),  # Convert to tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet statistics
            ])

        self.train_images = self.train_files['subDirectory_filePath']
        self.train_labels=self.train_files['expression']
        self.train_dataset = AffectNetDataset(self.train_images, self.train_labels, self.transform)

        self.val_images = self.val_files['subDirectory_filePath']
        self.val_labels=self.val_files['expression']
        self.val_dataset = AffectNetDataset(self.val_images, self.val_labels, self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=10)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=10)

    # def test_dataloader(self):
    #     return DataLoader(self.test_dataset, batch_size=10000, num_workers = self.num_workers)

# Testing the implementation
if __name__ == "__main__":
    # Initialize AffectNetDataset
    affectnet_data = AffectNetDataModule()
    
    # Get DataLoader
    train_loader = affectnet_data.train_dataloader()
    train_image, train_label = next(iter(train_loader))
    print(f"train_images: {train_image.shape}")
    print(f"train_labels: {train_label.shape}")