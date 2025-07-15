import torch
import numpy as np
from torchvision import models, transforms
from config import DEVICE, FRAME_SIZE

resnet = models.resnet18(pretrained=True).to(DEVICE).eval()

cnn_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(FRAME_SIZE),
    transforms.ToTensor()
])

def extract_cnn_features(image):
    img = cnn_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = resnet(img)
    return features.squeeze().cpu().numpy()
