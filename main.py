from dicom_loader import load_dicom_series
from visualizer import render_3d_model

if __name__ == "__main__":
    dicom_folder = "dicom_files/"  # Replace with your DICOM folder path
    output_image = "bounding_box_model.png"

    print("[🔍] Loading DICOM slices...")
    image_stack, spacing = load_dicom_series(dicom_folder)

    print("[⚙️] Rendering 3D model and generating bounding box image...")
    render_3d_model(image_stack, spacing, output_image)

    print(f"[✅] Bounding box image saved to: {output_image}")
