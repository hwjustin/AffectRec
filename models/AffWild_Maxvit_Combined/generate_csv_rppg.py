import pandas as pd
import numpy as np
import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
import torch.nn as nn
from PIL import Image

# Load the annotations for training and validation from separate CSV files
IMAGE_FOLDER_TEST = "dataset_new/cropped_aligned"
valid_annotations_path = (
    "dataset_new/csv_new/annotation_validation.csv"
)


valid_annotations_df = pd.read_csv(valid_annotations_path, dtype={"image": str})


# Set parameters
BATCHSIZE = 128
base_model = models.maxvit_t(weights="DEFAULT")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ValenceArousalModel(nn.Module):
    def __init__(self, base_model, rppg_dim=15):
        super(ValenceArousalModel, self).__init__()
        self.backbone = base_model  # Use the entire MaxViT model
        block_channels = base_model.classifier[3].in_features  # Number of features before the classifier
        self.backbone.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(block_channels),
            nn.Linear(block_channels, block_channels),
            nn.Tanh(),
            # nn.Linear(block_channels, 10, bias=False),
        )

        # rPPG branch
        self.rppg_branch = nn.Sequential(
            nn.Linear(rppg_dim, 64),
            nn.ReLU(),
            nn.Linear(64, block_channels),  # Match the backbone's output size
            nn.ReLU(),
        )

        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(block_channels * 2, 256),  # Combine image and rPPG features
            nn.ReLU(),
            nn.Linear(256, 10),  # 8 classes (expressions) + 2 regression outputs (valence, arousal)
        )

    def forward(self, image, rppg):
        # Extract features from the MaxViT backbone
        image_features = self.backbone(image) 
        # print("Pineapple", image_features.shape)
        # image_features = image_features.view(image_features.size(0), -1)  # Flatten

        # Process rPPG features
        # print("Pineapple", rppg)
        rppg_features = self.rppg_branch(rppg)
        # print("Pineapple1", rppg_features.shape)

        # Concatenate image and rPPG features
        combined_features = torch.cat((image_features, 0.3 * rppg_features), dim=1)

        # Final classifier
        outputs = self.classifier(combined_features)
        return outputs

# **** Create dataset and data loaders ****
class CustomDataset(Dataset):
    def __init__(self, dataframe, root_dir, rppg_dir, valid_expressions, transform=None, balance=False):
        self.transform = transform
        self.root_dir = root_dir
        self.rppg_dir = rppg_dir
        self.balance = balance
        self.count = 0

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
        video_folder = os.path.basename(os.path.dirname(image_path))  # Get subfolder
        rppg_path = os.path.join(self.rppg_dir, video_folder, "rppg_rppg.npz")
        if os.path.exists(rppg_path):
            rppg_data = np.load(rppg_path)["rppg"]  # Load rPPG from .npz file
            frame_index = self.dataframe.index[idx]
            start_idx = max(0, frame_index - 7)
            end_idx = min(len(rppg_data), frame_index + 8)
            rppg_feature = rppg_data[start_idx:end_idx]
            # Pad if the range is less than 31
            if len(rppg_feature) < 15:
                rppg_feature = np.pad(rppg_feature, (0, 15 - len(rppg_feature)), mode="constant")
        else:
            rppg_feature = np.zeros(15, dtype=np.float32)  # Handle missing rPPG

        rppg_feature = torch.tensor(rppg_feature, dtype=torch.float32)

        return image, rppg_feature, classes, labels
           

    def balance_dataset(self):
        balanced_df = self.dataframe.groupby("exp", group_keys=False).apply(
            lambda x: x.sample(self.dataframe["exp"].value_counts().min())
        )
        return balanced_df


transform_valid = transforms.Compose(
    [
        transforms.Resize((224, 224)), # resize every pitcure to 224*224
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

valid_dataset = CustomDataset(
    dataframe=valid_annotations_df,
    root_dir=IMAGE_FOLDER_TEST,
    rppg_dir="dataset_new/rppg",
    valid_expressions=[0, 1, 2, 3, 4, 5, 6, 7],
    transform=transform_valid,
    balance=False,
)
valid_loader = DataLoader(
    valid_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=48
)

# ***** Define the model *****

# Initialize the model
MODEL = ValenceArousalModel(base_model).to(DEVICE)
# block_channels = MODEL.classifier[3].in_features
# MODEL.classifier = nn.Sequential(
#     nn.AdaptiveAvgPool2d(1),
#     nn.Flatten(),
#     nn.LayerNorm(block_channels),
#     nn.Linear(block_channels, block_channels),
#     nn.Tanh(),
#     nn.Linear(block_channels, 10, bias=False),
# )
# MODEL.to(DEVICE)  # Put the model to the GPU

# Set the model to evaluation mode
MODEL.load_state_dict(torch.load("models/AffWild_Maxvit_Combined/model_best_rppg.pt"))
MODEL.to(DEVICE)
MODEL.eval()

all_labels_cls = []
all_predicted_cls = []

all_true_val = []
all_pred_val = []
all_true_aro = []
all_pred_aro = []

# Start inference on test set
with torch.no_grad():
    for images, rppgs, classes, labels in iter(valid_loader):
        images, rppgs, classes, labels = (
            images.to(DEVICE),
            rppgs.to(DEVICE),
            classes.to(DEVICE),
            labels.to(DEVICE),
        )

        outputs = MODEL(images, rppgs)
        outputs_cls = outputs[:, :8]
        outputs_reg = outputs[:, 8:]
        val_pred = outputs_reg[:, 0]
        aro_pred = outputs_reg[:, 1]

        _, predicted_cls = torch.max(outputs_cls, 1)

        all_labels_cls.extend(classes.cpu().numpy())
        all_predicted_cls.extend(predicted_cls.cpu().numpy())
        val_true = labels[:, 0]
        aro_true = labels[:, 1]

        all_true_val.extend(val_true.cpu().numpy())
        all_true_aro.extend(aro_true.cpu().numpy())
        all_pred_val.extend(val_pred.cpu().numpy())
        all_pred_aro.extend(aro_pred.cpu().numpy())

df = pd.DataFrame(
    {
        "cat_pred": all_predicted_cls,
        "cat_true": all_labels_cls,
        "val_pred": all_pred_val,
        "val_true": all_true_val,
        "aro_pred": all_pred_aro,
        "aro_true": all_true_aro,
    }
)
df.to_csv("inference.csv", index=False)
