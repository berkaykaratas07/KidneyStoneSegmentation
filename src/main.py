# main.py
import os
import torch
from segmentation import load_sam_model, show_anns
from utils import process_images


# Step 1: Set up initial parameters and environment
data_dir = "data"
sam_checkpoint = "C:\\Users\\Berkay\\PycharmProjects\\KidneyStone01\\sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Step 2: Load SAM model and initialize predictor
predictor = load_sam_model(model_type, sam_checkpoint, device)

# Step 3: Process train, test, and validation sets with a limit of 5 images per set
for split in ['train', 'test', 'valid']:
    image_dir = os.path.join(data_dir, split, "images")
    label_dir = os.path.join(data_dir, split, "labels")

    if os.path.exists(image_dir) and os.path.exists(label_dir):
        process_images(predictor, image_dir, label_dir, device, limit=5)
    else:
        print(f"'{image_dir}' or '{label_dir}' not found for {split} set.")