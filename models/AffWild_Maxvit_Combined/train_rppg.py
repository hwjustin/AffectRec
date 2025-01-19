import pandas as pd
import numpy as np
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
IMAGE_FOLDER = "dataset_new/cropped_aligned/"
IMAGE_FOLDER_TEST = "dataset_new/cropped_aligned/"
train_annotations_path = (
    "dataset_new/csv_new/annotation_train.csv"
)
valid_annotations_path = (
    "dataset_new/csv_new/annotation_validation.csv"
)
train_annotations_df = pd.read_csv(train_annotations_path, dtype={"image": str})
valid_annotations_df = pd.read_csv(valid_annotations_path, dtype={"image": str})


# Set parameters
BATCHSIZE = 64 # original batch size is 128, CUDA out of memory for P100
NUM_EPOCHS = 20
LR = 4e-5
base_model = models.maxvit_t(weights="DEFAULT")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ValenceArousalModel(nn.Module):
    def __init__(self, base_model, rppg_dim=31):
        super(ValenceArousalModel, self).__init__()
        self.backbone = base_model 
        block_channels = base_model.classifier[3].in_features  
        self.backbone.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(block_channels),
            nn.Linear(block_channels, block_channels),
            nn.Tanh(),
        )

        self.rppg_branch = nn.Sequential(
            nn.Linear(rppg_dim, 64),
            nn.ReLU(),
            nn.Linear(64, block_channels), 
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(block_channels * 2, 256),  
            nn.ReLU(),
            nn.Linear(256, 10), 
        )

    def forward(self, image, rppg):
        image_features = self.backbone(image) 

        rppg_features = self.rppg_branch(rppg)
        combined_features = torch.cat((image_features, 0.3 * rppg_features), dim=1)
        
        outputs = self.classifier(combined_features)
        return outputs

# **** Create dataset and data loaders ****
class CustomDataset(Dataset):
    def __init__(self, dataframe, root_dir, rppg_dir, valid_expressions, transform=None, balance=False):
        self.transform = transform
        self.root_dir = root_dir
        self.rppg_dir = rppg_dir
        self.balance = balance

        # filter out invalid expressions
        if valid_expressions is not None:
            dataframe = dataframe[dataframe["expression"].isin(valid_expressions)]
        
        dataframe = dataframe[
            (dataframe["valence"] >= -1) & (dataframe["valence"] <= 1) &
            (dataframe["arousal"] >= -1) & (dataframe["arousal"] <= 1)
        ]

        self.dataframe = dataframe

        if self.balance:
            self.dataframe = self.balance_dataset()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = os.path.join(
            self.root_dir, f"{self.dataframe['image'].iloc[idx]}"
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
        
         # rPPG feature extraction
        video_folder = os.path.basename(os.path.dirname(image_path)) 
        rppg_path = os.path.join(self.rppg_dir, video_folder, "rppg_rppg.npz")
        if os.path.exists(rppg_path):
            rppg_data = np.load(rppg_path)["rppg"] 
            frame_index = self.dataframe.index[idx]
            start_idx = max(0, frame_index - 15)
            end_idx = min(len(rppg_data), frame_index + 16)
            rppg_feature = rppg_data[start_idx:end_idx]
            if len(rppg_feature) < 31:
                rppg_feature = np.pad(rppg_feature, (0, 31 - len(rppg_feature)), mode="constant")
        else:
            rppg_feature = np.zeros(31, dtype=np.float32)

        rppg_feature = torch.tensor(rppg_feature, dtype=torch.float32)

        return image, rppg_feature, classes, labels

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
    rppg_dir="dataset_new/rppg",
    valid_expressions=[0, 1, 2, 3, 4, 5, 6, 7],
    transform=transform,
    balance=False,
)
valid_dataset = CustomDataset(
    dataframe=valid_annotations_df,
    root_dir=IMAGE_FOLDER_TEST,
    rppg_dir="dataset_new/rppg",
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
MODEL = ValenceArousalModel(base_model).to(DEVICE)

# Define (weighted) loss function
weights = torch.tensor(
    [0.0448, 0.1026, 0.4630, 0.1551, 0.0362, 0.0621, 0.1035, 0.0329]
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
    for images, rppgs, classes, labels in tqdm(
        train_loader, desc="Epoch train_loader progress"
    ):
        images, rppgs, classes, labels = (
            images.to(DEVICE),
            rppgs.to(DEVICE),
            classes.to(DEVICE),
            labels.to(DEVICE),
        )
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = MODEL(images, rppgs)
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
        for images, rppgs, classes, labels in valid_loader:
            images, rppgs, classes, labels = (
                images.to(DEVICE),
                rppgs.to(DEVICE),
                classes.to(DEVICE),
                labels.to(DEVICE),
            )
            outputs = MODEL(images, rppgs)
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
