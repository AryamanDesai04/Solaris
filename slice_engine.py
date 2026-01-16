import os
import pydicom
import imageio
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def load_dicom_slices(folder_path):
    slices = []
    for filename in sorted(glob(os.path.join(folder_path, "*.dcm"))):
        dicom_data = pydicom.dcmread(filename)
        if hasattr(dicom_data, 'pixel_array'):
            slices.append(dicom_data.pixel_array)
    return slices

def normalize(slice_data):
    return (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))

def generate_slices_and_gif(dicom_folder):
    slices = load_dicom_slices(dicom_folder)
    images = []

    gif_output_path = os.path.join("static/gif", "slices.gif")

    for i, slice in enumerate(slices):
        norm_slice = normalize(slice)
        image_path = os.path.join("static/gif", f"slice_{i}.png")
        plt.imsave(image_path, norm_slice, cmap='gray')
        images.append(imageio.imread(image_path))

    imageio.mimsave(gif_output_path, images, duration=0.1)
    return slices, gif_output_path
