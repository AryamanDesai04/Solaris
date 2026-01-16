import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import imageio

def load_dicom_series(folder_path):
    slices = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".dcm"):
            ds = pydicom.dcmread(os.path.join(folder_path, filename))
            slices.append(ds)

    if not slices:
        raise ValueError("No DICOM files found.")

    slices.sort(key=lambda x: int(x.InstanceNumber) if hasattr(x, 'InstanceNumber') else 0)
    volume = np.stack([s.pixel_array for s in slices])
    return volume

def visualize_dicom_stack(folder_path, output_folder):
    volume = load_dicom_series(folder_path)
    gif_path = os.path.join(output_folder, 'volume_preview.gif')

    # Optional: Clear old frames
    for file in os.listdir(output_folder):
        if file.endswith('.png') or file.endswith('.gif'):
            os.remove(os.path.join(output_folder, file))

    # Generate 2D slices + bounding box
    images = []
    for i in range(volume.shape[0]):
        fig, ax = plt.subplots()
        ax.imshow(volume[i], cmap='gray')
        ax.set_title(f'Slice {i}')
        rect = plt.Rectangle((10, 10), volume.shape[1]-20, volume.shape[2]-20,
                             linewidth=1.5, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        plt.axis('off')
        temp_path = os.path.join(output_folder, f'slice_{i}.png')
        plt.savefig(temp_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        images.append(imageio.imread(temp_path))

    imageio.mimsave(gif_path, images, duration=0.3)
    return 'volume_preview.gif'
