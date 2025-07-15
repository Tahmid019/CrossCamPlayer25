import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DETECTION_CONF = 0.3
FRAME_SIZE = (224, 224)
OUTPUT_JSON = "output/player_mapping.json"
