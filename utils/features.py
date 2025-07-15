import torch
import numpy as np
from torchvision import models, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
resnet = models.resnet18(pretrained=True).to(device).eval()

cnn_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def extract_cnn_features(image):
    img = cnn_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = resnet(img)
    return features.squeeze().cpu().numpy()
