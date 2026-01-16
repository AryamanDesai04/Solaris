import os
import imageio
import pydicom
import numpy as np
import cv2

def generate_gif_from_dicom(dicom_folder, output_path):
    images = []
    files = sorted([f for f in os.listdir(dicom_folder) if f.lower().endswith('.dcm')])
    
    for filename in files:
        path = os.path.join(dicom_folder, filename)
        try:
            ds = pydicom.dcmread(path, force=True)
            img = ds.pixel_array.astype(np.float32)
            # Normalize to 0-255 uint8
            img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            img_rgb = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2RGB)
            images.append(img_rgb)
        except Exception as e:
            print(f"[GIF] skipping {path}: {e}")
    
    if not images:
        raise RuntimeError("No valid DICOM images found to generate GIF.")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, images, fps=5)
