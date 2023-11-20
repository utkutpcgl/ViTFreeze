import numpy as np
from PIL import Image
import random

def patchify_and_save_image(image_path, save_path, patch_size=14, mask_percentage=0.75):
    # Load the image
    image = Image.open(image_path)
    image = np.array(image)

    # Calculate the number of patches
    rows, cols, _ = image.shape
    num_patches_x = cols // patch_size
    num_patches_y = rows // patch_size

    # Create a list of all patches
    all_patches = [(x, y) for x in range(num_patches_x) for y in range(num_patches_y)]

    # Randomly select 75% of the patches to mask
    num_patches_to_mask = int(len(all_patches) * mask_percentage)
    patches_to_mask = random.sample(all_patches, num_patches_to_mask)

    # Mask the selected patches
    for x, y in patches_to_mask:
        image[y*patch_size:(y+1)*patch_size, x*patch_size:(x+1)*patch_size] = 0

    # Convert back to PIL Image
    patched_image = Image.fromarray(image)

    # Save the patched image
    patched_image.save(save_path)

# Usage example
patchify_and_save_image("imagenet_anaconda.JPEG", "imagenet_anaconda_patch.JPEG")
