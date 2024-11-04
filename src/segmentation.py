import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor


def load_sam_model(model_type, checkpoint, device):
    """Load the SAM model and initialize the predictor."""
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor


def show_anns(masks):
    """Display segmentation masks with transparency."""
    if len(masks) == 0:
        return

    # Sort masks by area (largest first)
    sorted_masks = sorted(masks, key=lambda x: x.sum(), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    # Initialize an RGBA image
    img = np.ones((sorted_masks[0].shape[1], sorted_masks[0].shape[2], 4))
    img[:, :, 3] = 0  # Set alpha channel to 0 for full transparency

    for mask in sorted_masks:
        if not isinstance(mask, np.ndarray):
            mask = mask.cpu().numpy()  # Convert tensor to numpy array if needed
        mask = np.squeeze(mask)  # Remove any extra dimensions

        # Generate a random color for each mask with semi-transparency
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[mask.astype(bool)] = color_mask  # Apply color to mask area in the image

    ax.imshow(img)
