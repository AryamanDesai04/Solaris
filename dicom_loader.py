import os
import pydicom
import numpy as np

def load_dicom_images(folder_path):
    slices = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".dcm"):
            ds = pydicom.dcmread(os.path.join(folder_path, filename))
            slices.append(ds.pixel_array)

    volume = np.stack(slices).astype(np.int16)
    return volume
