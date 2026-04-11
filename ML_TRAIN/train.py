import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import scipy.io as sio


# -------------------------------
# Model Architecture (MC-CNN)
# -------------------------------

class MC_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.column1 = nn.Sequential(
            nn.Conv2d(3, 8, 9, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 7, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 7, padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 16, 7, padding='same'),
            nn.ReLU(),
            nn.Conv2d(16, 8, 7, padding='same'),
            nn.ReLU(),
        )

        self.column2 = nn.Sequential(
            nn.Conv2d(3, 10, 7, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(10, 20, 5, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 40, 5, padding='same'),
            nn.ReLU(),
            nn.Conv2d(40, 20, 5, padding='same'),
            nn.ReLU(),
            nn.Conv2d(20, 10, 5, padding='same'),
            nn.ReLU(),
        )

        self.column3 = nn.Sequential(
            nn.Conv2d(3, 12, 5, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(12, 24, 3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(48, 24, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(24, 12, 3, padding='same'),
            nn.ReLU(),
        )

        self.fusion_layer = nn.Conv2d(30, 1, 1)

    def forward(self, x):
        x1 = self.column1(x)
        x2 = self.column2(x)
        x3 = self.column3(x)
        x = torch.cat((x1, x2, x3), dim=1)
        return self.fusion_layer(x)


class CrowdDataset(Dataset):
    """
    Expects:
      images/        -> crowd images (.jpg / .png)
      density_maps/  -> ground-truth density maps (.mat files)
    """

    def __init__(self, image_dir, density_dir):
        self.image_dir = image_dir
        self.density_dir = density_dir
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)

        mat_name = img_name.replace(".jpg", ".mat")
        mat_path = os.path.join(self.density_dir, mat_name)

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1)) / 255.0
        image = torch.tensor(image, dtype=torch.float32)

        mat = sio.loadmat(mat_path)

        if "density" in mat:
            density_map = mat["density"]
        elif "ground_truth" in mat:
            density_map = mat["ground_truth"]
        else:
            density_map = list(mat.values())[-1]

        density_map = torch.tensor(density_map, dtype=torch.float32).unsqueeze(0)

        return image, density_map

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MC_CNN().to(device)
    criterion = nn.MSELoss()  # Density map regression loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    dataset = CrowdDataset(
        image_dir="data/images",
        density_dir="data/density_maps"
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    epochs = 50
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for images, density_maps in tqdm(dataloader):
            images = images.to(device)
            density_maps = density_maps.to(device)

            preds = model(images)
            loss = criterion(preds, density_maps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), "crowd_counting.pth")
    print("Model training complete.")


if __name__ == "__main__":
    train()
