# utils.py
import os
import cv2
import torch
from segmentation import show_anns
import matplotlib.pyplot as plt


def process_images(predictor, image_dir, label_dir, device, limit=5):
    """Process images and perform segmentation with a limit on number of images."""
    processed_images = 0

    for filename in os.listdir(image_dir):
        if processed_images >= limit:
            break
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Support image file types
            image_path = os.path.join(image_dir, filename)
            label_path = os.path.join(label_dir, filename.replace(".jpg", ".txt").replace(".png", ".txt"))

            # Load and prepare the image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image)

            # Load labels and create bounding boxes
            input_boxes = []
            with open(label_path, 'r') as f:
                for line in f:
                    values = line.strip().split()
                    _, x_center, y_center, width, height = map(float, values)
                    image_height, image_width, _ = image.shape
                    x1 = int((x_center - width / 2) * image_width)
                    y1 = int((y_center - height / 2) * image_height)
                    x2 = int((x_center + width / 2) * image_width)
                    y2 = int((y_center + height / 2) * image_height)
                    input_boxes.append([x1, y1, x2, y2])

            # Perform segmentation with SAM
            transformed_boxes = predictor.transform.apply_boxes_torch(torch.tensor(input_boxes), image.shape[:2]).to(
                device)
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )

            # Visualize the results
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            show_anns(masks.cpu().numpy())
            plt.title(f"Segmentation - {filename}")
            plt.axis('off')
            plt.show()

            processed_images += 1
