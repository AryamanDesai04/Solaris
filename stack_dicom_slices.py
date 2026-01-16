import numpy as np
from PIL import Image
import os

def stack_slices_to_image(volume):
    norm_volume = (volume - volume.min()) / (volume.max() - volume.min())
    norm_volume = (norm_volume * 255).astype(np.uint8)

    selected_slices = norm_volume[::max(1, len(norm_volume) // 10)]
    images = [Image.fromarray(slice_) for slice_ in selected_slices]
    stacked = np.hstack(images)

    output_path = 'static/output.png'
    os.makedirs('static', exist_ok=True)
    Image.fromarray(stacked).save(output_path)

    return output_path
