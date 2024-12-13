import pandas as pd
import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.optim import lr_scheduler
from tqdm import tqdm

# Load the annotations for training and validation from separate CSV files
IMAGE_FOLDER = "dataset_new/cropped_aligned"
IMAGE_FOLDER_TEST = "dataset_new/cropped_aligned"
train_annotations_path = (
    "dataset_new/csv/annotations_train.csv"
)
valid_annotations_path = (
    "dataset_new/csv/annotations_validation.csv"
)
train_annotations_df = pd.read_csv(train_annotations_path)
valid_annotations_df = pd.read_csv(valid_annotations_path)


# Set parameters
BATCHSIZE = 128 # original batch size is 128, CUDA out of memory for P100
NUM_EPOCHS = 20
LR = 4e-5
MODEL = models.maxvit_t(weights="DEFAULT")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# **** Create dataset and data loaders ****
class CustomDataset(Dataset):
    def __init__(self, dataframe, root_dir, valid_expressions, transform=None, balance=False):
        self.transform = transform
        self.root_dir = root_dir
        self.balance = balance

        # filter out invalid expressions
        if valid_expressions is not None:
            dataframe = dataframe[dataframe["expression"].isin(valid_expressions)]
        self.dataframe = dataframe

        if self.balance:
            self.dataframe = self.balance_dataset()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = os.path.join(
            self.root_dir, f"{self.dataframe['filename'].iloc[idx]}", f"{self.dataframe['index'].iloc[idx]}.jpg"
        )
        if os.path.exists(image_path):
            image = Image.open(image_path)
        else:
            image = Image.new(
                "RGB", (224, 224), color="white"
            )  # Handle missing image file

        classes = torch.tensor(self.dataframe["expression"].iloc[idx], dtype=torch.long)
        labels = torch.tensor(self.dataframe[['valence', 'arousal']].iloc[idx].values, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, classes, labels

    def balance_dataset(self):
        balanced_df = self.dataframe.groupby("exp", group_keys=False).apply(
            lambda x: x.sample(self.dataframe["exp"].value_counts().min())
        )
        return balanced_df


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)), # resize every pitcure to 224*224
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomGrayscale(0.01),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        ),  # model more robust to changes in lighting conditions.
        transforms.RandomPerspective(
            distortion_scale=0.2, p=0.5
        ),  # can be helpful if your images might have varying perspectives.
        
        transforms.ToTensor(),  # saves image as tensor (automatically divides by 255)
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(
            p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random"
        ),  # TEST: Should help overfitting
    ]
)

transform_valid = transforms.Compose(
    [
        transforms.Resize((224, 224)), # resize every pitcure to 224*224
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

train_dataset = CustomDataset(
    dataframe=train_annotations_df,
    root_dir=IMAGE_FOLDER,
    valid_expressions=[0, 1, 2, 3, 4, 5, 6, 7],
    transform=transform,
    balance=False,
)
valid_dataset = CustomDataset(
    dataframe=valid_annotations_df,
    root_dir=IMAGE_FOLDER_TEST,
    valid_expressions=[0, 1, 2, 3, 4, 5, 6, 7],
    transform=transform_valid,
    balance=False,
)
train_loader = DataLoader(
    train_dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=32
)
valid_loader = DataLoader(
    valid_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=32
)

# ***** Define the model *****

# Initialize the model
block_channels = MODEL.classifier[3].in_features
MODEL.classifier = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.LayerNorm(block_channels),
    nn.Linear(block_channels, block_channels),
    nn.Tanh(),
    nn.Linear(block_channels, 10, bias=False),
)

MODEL.to(DEVICE)  # Put the model to the GPU

# Define (weighted) loss function
weights = torch.tensor(
    [0.015605, 0.008709, 0.046078, 0.083078, 0.185434, 0.305953, 0.046934, 0.30821]
)
criterion_cls = nn.CrossEntropyLoss(weights.to(DEVICE))
criterion_cls_val = (
    nn.CrossEntropyLoss()
)  # Use two loss functions, as the validation dataset is balanced
criterion_reg = nn.MSELoss()

optimizer = optim.AdamW(MODEL.parameters(), lr=LR)
lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=BATCHSIZE * NUM_EPOCHS)

# ***** Train the model *****
print("--- Start training ---")
scaler = torch.cuda.amp.GradScaler()
best_valid_loss = 100

for epoch in range(NUM_EPOCHS):
    MODEL.train()
    total_train_correct = 0
    total_train_samples = 0
    for images, classes, labels in tqdm(
        train_loader, desc="Epoch train_loader progress"
    ):
        images, classes, labels = (
            images.to(DEVICE),
            classes.to(DEVICE),
            labels.to(DEVICE),
        )
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = MODEL(images)
            outputs_cls = outputs[:, :8]
            outputs_reg = outputs[:, 8:]
            loss = criterion_cls(
                outputs_cls.cuda(), classes.cuda()
            ) + 5 * criterion_reg(outputs_reg.cuda(), labels.cuda())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

        _, train_predicted = torch.max(outputs_cls, 1)
        total_train_samples += classes.size(0)
        total_train_correct += (train_predicted == classes).sum().item()

    train_accuracy = (total_train_correct / total_train_samples) * 100

    MODEL.eval()
    valid_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, classes, labels in valid_loader:
            images, classes, labels = (
                images.to(DEVICE),
                classes.to(DEVICE),
                labels.to(DEVICE),
            )
            outputs = MODEL(images)
            outputs_cls = outputs[:, :8]
            outputs_reg = outputs[:, 8:]
            loss = criterion_cls_val(
                outputs_cls.cuda(), classes.cuda()
            ) + 5 * criterion_reg(outputs_reg.cuda(), labels.cuda())
            valid_loss += loss.item()
            _, predicted = torch.max(outputs_cls, 1)
            total += classes.size(0)
            correct += (predicted == classes).sum().item()

    print(
        f"Epoch [{epoch+1}/{NUM_EPOCHS}] - "
        f"Validation Loss: {valid_loss/len(valid_loader):.4f}, "
        f"Validation Accuracy: {(correct/total)*100:.2f}%"
        f", Training Accuracy: {train_accuracy:.2f}%, "
    )

    avg_valid_loss = valid_loss/len(valid_loader)

    if avg_valid_loss < best_valid_loss:
        best_valid_loss = avg_valid_loss
        print(f"epoch {epoch+1} generates better result!")
    
    print(f"Saving model at epoch {epoch+1}")
    torch.save(MODEL.state_dict(), f"results/model_epoch{epoch+1}.pt")  # Save the best model
