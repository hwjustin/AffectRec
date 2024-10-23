import pandas as pd
import torch
from enum import Enum 

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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

class ToyDataset(Dataset):
	def __init__(self, data, labels):
		self.labels = labels
		self.data = data

	def __len__(self):
		return self.labels.shape[0]

	def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]

class AffectNetDataset(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.train_files = pd.read_csv("dataset/filelists/training.csv") # shuffle if needed: .sample(frac=1, random_state=27)
        self.val_files = pd.read_csv("dataset/filelists/validation.csv", header=None) # might need to figure out later
        header = self.train_files.columns
        self.val_files.columns = header

        self.train_images = self.train_files['subDirectory_filePath']
        self.train_labels=self.train_files['expression']
        self.train_dataset = ToyDataset(self.train_images, self.train_labels)

        self.val_images = self.val_files['subDirectory_filePath']
        self.val_labels=self.val_files['expression']
        self.val_dataset = ToyDataset(self.val_images, self.val_labels)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=10)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=10)

    # def test_dataloader(self):
    #     return DataLoader(self.test_dataset, batch_size=10000, num_workers = self.num_workers)

# Testing the implementation
if __name__ == "__main__":
    # Initialize AffectNetDataset
    affectnet_data = AffectNetDataset()
    
    # Get DataLoader
    train_loader = affectnet_data.val_dataloader()
    train_image, train_label = next(iter(train_loader))
    print(f"train_images: {train_image}")
    print(f"train_labels: {train_label}")