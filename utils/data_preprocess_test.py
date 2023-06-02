import numpy as np
from PIL import Image
import math

def preprocess(mask_values, pil_img, scale=1.0, is_mask=False):
    resized_img = np.array(pil_img)
    oldH, oldW = resized_img.shape[:2]

    # Resize image/mask while preserving aspect ratio
    newH = math.ceil(scale * oldH)
    newW = math.ceil(scale * oldW)
    aspect_ratio = oldW / oldH
    if aspect_ratio > 1:
        newW = math.ceil(scale * oldW / oldH * newH)
    else:
        newH = math.ceil(scale * oldH / oldW * newW)
    resized_img = pil_img.resize((newW, newH))

    if is_mask:
        mask = np.array(pil_img.resize((newW, newH)))
        mask_indices = np.zeros((newW, newH), dtype=np.int64)

        for i, v in enumerate(mask_values):
            mask_indices[resized_img == v] = i
        return mask_indices

    else:
        if not isinstance(resized_img, np.ndarray):
            resized_img = np.array(resized_img)
        if resized_img.ndim == 2:
            resized_img = resized_img[np.newaxis, ...]
        else:
            resized_img = resized_img.transpose((2, 0, 1))

        if (resized_img > 1).any():
            resized_img = resized_img / 255.0

        return resized_img


# Example usage with different sized images and masks
img_path = "example_image.jpg"
mask_path = "example_mask.jpg"

# Load image and mask using PIL
img = Image.open(img_path)
mask = Image.open(mask_path)

# Define unique mask values
mask_values = [0, 1]

# Resize and preprocess image and mask
preprocessed_img = preprocess(None, img, scale=0.5, is_mask=False)
preprocessed_mask = preprocess(mask_values, mask, scale=0.5, is_mask=True)

preprocessed_img = (preprocessed_img * 255.0).astype(np.uint8)
preprocessed_mask = (preprocessed_mask * 255.0).astype(np.uint8)

# Save preprocessed images and masks
preprocessed_img = Image.fromarray(preprocessed_img)
preprocessed_mask = Image.fromarray(preprocessed_mask)

preprocessed_img.save("preprocessed_image.jpg")
preprocessed_mask.save("preprocessed_mask.jpg")