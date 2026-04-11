from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app) # Enable CORS for frontend requests

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

        self.fusion_layer = nn.Sequential(
            nn.Conv2d(30, 1, 1, padding=0),
        )

    def forward(self, img_tensor):
        x1 = self.column1(img_tensor)
        x2 = self.column2(img_tensor)
        x3 = self.column3(img_tensor)
        x = torch.cat((x1, x2, x3), 1)
        x = self.fusion_layer(x)
        return x


def preprocess_image_from_bytes(image_bytes, gt_downsample=4):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ds_rows = int(img.shape[0] // gt_downsample)
    ds_cols = int(img.shape[1] // gt_downsample)
    img = cv2.resize(img, (ds_cols * gt_downsample, ds_rows * gt_downsample))
    img = img.transpose((2, 0, 1))
    img_tensor = torch.tensor(img / 255, dtype=torch.float)
    return img_tensor.unsqueeze(0)


def density_map_to_base64(density_map_np):
    plt.figure(figsize=(4, 4))
    plt.imshow(density_map_np, cmap='jet')
    plt.axis("off")

    buffer = BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0)
    plt.close()

    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return img_base64


def allocate_resources(pred_count, density_map):
    security_guard_thresholds = {50: 1, 100: 2, 200: 3, float('inf'): 4}
    num_security_guards = next(val for key, val in security_guard_thresholds.items() if pred_count <= key)

    density_map_np = density_map.squeeze().detach().cpu().numpy()
    rows, cols = density_map_np.shape

    north_density = float(np.sum(density_map_np[:rows // 2, :]))
    south_density = float(np.sum(density_map_np[rows // 2:, :]))
    west_density = float(np.sum(density_map_np[:, :cols // 2]))
    east_density = float(np.sum(density_map_np[:, cols // 2:]))

    directions = {"North": north_density, "South": south_density, "West": west_density, "East": east_density}
    max_density_value = max(directions.values())
    max_density_direction = [k for k, v in directions.items() if v == max_density_value]

    heatmap_base64 = density_map_to_base64(density_map_np)

    return num_security_guards, max_density_direction, directions, heatmap_base64


model = MC_CNN()
try:
    model.load_state_dict(torch.load("crowd_counting.pth", map_location=torch.device('cpu')), strict=False)
except Exception as e:
    print("Warning: Could not load model parameters. Expected relative path `crowd_counting.pth` ->", str(e))
model.eval()


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    img_tensor = preprocess_image_from_bytes(file.read())

    with torch.inference_mode():
        pred_dm = model(img_tensor)

    pred_count = int(np.maximum(0, pred_dm.sum().item()))

    guards, high_density_dirs, all_dirs, heatmap_img = allocate_resources(pred_count, pred_dm)

    return jsonify({
        "predicted_count": pred_count,
        "security_guards": guards,
        "high_density_direction": high_density_dirs,
        "directional_density": all_dirs,
        "density_map_image": heatmap_img   
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(port=8000, host='0.0.0.0')
