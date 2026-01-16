import numpy as np

def preprocess_volume(volume, threshold=100):
    """
    Convert the 3D volume to a binary mask where foreground = 1, background = 0.
    """
    binary_volume = (volume > threshold).astype(np.uint8)
    return binary_volume
