# extract_engine.py
import os
import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError
from skimage import measure, filters, morphology
from scipy import ndimage as ndi
from scipy.ndimage import zoom
import plotly.graph_objects as go
import plotly.io as pio

# Optional STL export
try:
    from stl import mesh as stl_mesh
    _HAS_STL = True
except Exception:
    _HAS_STL = False
    stl_mesh = None


# ---- Helpers ----
def read_dicom_datasets(folder_path):
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.dcm')])
    datasets = []
    for f in files:
        path = os.path.join(folder_path, f)
        try:
            ds = pydicom.dcmread(path, force=True)
            # only keep if pixel data available
            if hasattr(ds, 'pixel_array') or 'PixelData' in ds:
                datasets.append(ds)
        except Exception as e:
            print(f"[read] skipping {path}: {e}")
    if not datasets:
        raise RuntimeError("No valid DICOM datasets found")
    return datasets


def get_spacing_and_sort(datasets):
    zpos = []
    for ds in datasets:
        try:
            zpos.append(float(ds.ImagePositionPatient[2]))
        except Exception:
            zpos.append(None)

    paired = list(zip(datasets, zpos))
    # if any z present, sort by it; otherwise try InstanceNumber fallback
    if any(z is not None for _, z in paired):
        paired = sorted(enumerate(paired),
                        key=lambda ip: (ip[1][1] if ip[1][1] is not None else float('inf'), ip[0]))
        datasets_sorted = [p[1][0] for p in paired]
        z_values = [p[1][1] for p in paired]
    else:
        try:
            datasets_sorted = sorted(datasets, key=lambda ds: int(ds.InstanceNumber))
        except Exception:
            datasets_sorted = datasets
        z_values = list(range(len(datasets_sorted)))

    # pixel spacing
    try:
        ps = datasets_sorted[0].PixelSpacing
        py = float(ps[0])
        px = float(ps[1])
    except Exception:
        px = py = 1.0

    # slice thickness / spacing in z
    pz = None
    try:
        if hasattr(datasets_sorted[0], 'SliceThickness'):
            pz = float(datasets_sorted[0].SliceThickness)
    except Exception:
        pz = None

    if pz is None and len(z_values) >= 2 and z_values[0] is not None:
        diffs = np.abs(np.diff([z for z in z_values if z is not None]))
        pz = float(np.median(diffs)) if len(diffs) > 0 else 1.0
    if pz is None:
        pz = 1.0

    return datasets_sorted, (pz, py, px)


def to_hounsfield(ds):
    arr = ds.pixel_array.astype(np.float32)
    slope = float(ds.get('RescaleSlope', 1.0))
    intercept = float(ds.get('RescaleIntercept', 0.0))
    return arr * slope + intercept


def stack_volume(datasets, prefer_hu=True):
    stacks = []
    for ds in datasets:
        try:
            if prefer_hu:
                stacks.append(to_hounsfield(ds))
            else:
                stacks.append(ds.pixel_array.astype(np.float32))
        except Exception:
            stacks.append(ds.pixel_array.astype(np.float32))
    vol = np.stack(stacks, axis=0)
    return vol


def window_normalize(volume, window_center=None, window_width=None):
    if window_center is not None and window_width is not None:
        low = window_center - window_width / 2.0
        high = window_center + window_width / 2.0
        vol = np.clip(volume, low, high)
        vol = (vol - low) / (high - low)
        vol = np.clip(vol, 0.0, 1.0)
    else:
        vmin = np.nanmin(volume)
        vmax = np.nanmax(volume)
        if vmax == vmin:
            vol = np.zeros_like(volume)
        else:
            vol = (volume - vmin) / (vmax - vmin)
    return vol.astype(np.float32)


def make_binary_mask(volume_norm, method='otsu', manual_threshold=None, pct=0.5):
    if method == 'manual' and manual_threshold is not None:
        th = manual_threshold
    elif method == 'percentile':
        th = np.percentile(volume_norm, float(pct * 100))
    else:
        try:
            # use central half for robustness
            sample = volume_norm[max(0, volume_norm.shape[0]//4):min(volume_norm.shape[0], 3*volume_norm.shape[0]//4)]
            flat = sample.flatten()
            flat = flat[np.isfinite(flat)]
            if flat.size == 0:
                th = 0.5
            else:
                th = filters.threshold_otsu(flat)
        except Exception:
            th = float(np.mean(volume_norm))
    mask = (volume_norm >= th).astype(np.uint8)
    return mask, float(th)


def clean_mask(mask, min_size_fraction=0.001, closing_iter=2):
    struct = morphology.ball(1)
    clean = ndi.binary_closing(mask, structure=struct, iterations=closing_iter)
    labeled, n = ndi.label(clean)
    if n == 0:
        return clean.astype(np.uint8)
    counts = np.bincount(labeled.flatten())
    counts[0] = 0
    largest_label = int(np.argmax(counts))
    final = np.zeros_like(mask, dtype=np.uint8)
    total_voxels = mask.size
    min_voxels = max(100, int(total_voxels * min_size_fraction))
    for lab in range(1, n+1):
        comp = (labeled == lab)
        if comp.sum() >= min_voxels or lab == largest_label:
            final = np.logical_or(final, comp)
    # fill holes slice-wise
    for z in range(final.shape[0]):
        slice_filled = ndi.binary_fill_holes(final[z])
        final[z] = slice_filled
    return final.astype(np.uint8)


def crop_to_bbox(volume, mask, pad=2):
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return volume, mask, (0, 0, 0)
    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0)
    zmin = max(0, zmin - pad)
    ymin = max(0, ymin - pad)
    xmin = max(0, xmin - pad)
    zmax = min(volume.shape[0]-1, zmax + pad)
    ymax = min(volume.shape[1]-1, ymax + pad)
    xmax = min(volume.shape[2]-1, xmax + pad)
    vol_c = volume[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]
    mask_c = mask[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]
    return vol_c, mask_c, (int(zmin), int(ymin), int(xmin))


def compute_mesh_metrics(verts, faces):
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    tri_vec = np.cross(v1 - v0, v2 - v0)
    tri_area = 0.5 * np.linalg.norm(tri_vec, axis=1)
    surface_area_mm2 = float(np.sum(tri_area))
    return {
        "surface_area_mm2": surface_area_mm2,
        "n_vertices": int(len(verts)),
        "n_faces": int(len(faces))
    }


def extract_mesh_and_export(volume, spacing, plotly_path, stl_path=None, threshold=None):
    """
    Extract mesh with marching cubes, save a Plotly HTML and optional STL.
    Returns mesh_metrics (dict) and used threshold (float).
    """
    if threshold is None:
        threshold = float(np.mean(volume))
    # marching_cubes expects the volume in same units as threshold
    verts, faces, normals, values = measure.marching_cubes(volume, level=threshold, spacing=spacing)
    x, y, z = verts.T
    i, j, k = faces.T.astype(int)
    mesh3d = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='lightblue', opacity=0.7)
    fig = go.Figure(data=[mesh3d])
    fig.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, b=0, t=0))
    os.makedirs(os.path.dirname(plotly_path), exist_ok=True)
    pio.write_html(fig, file=plotly_path, auto_open=False)

    metrics = compute_mesh_metrics(verts, faces)

    # optional STL export
    if stl_path:
        if not _HAS_STL:
            print("[STL] numpy-stl not installed; skipping STL export")
            stl_written = None
        else:
            try:
                faces_tri = faces.astype(np.int32)
                vertices = verts.astype(np.float32)
                n_faces = faces_tri.shape[0]
                data = np.zeros(n_faces, dtype=stl_mesh.Mesh.dtype)
                for idx in range(n_faces):
                    for jdx in range(3):
                        data['vectors'][idx][jdx] = vertices[faces_tri[idx, jdx], :]
                m = stl_mesh.Mesh(data.copy())
                os.makedirs(os.path.dirname(stl_path), exist_ok=True)
                m.save(stl_path)
                stl_written = stl_path
            except Exception as e:
                print(f"[STL] failed to save: {e}")
                stl_written = None
    else:
        stl_written = None

    return metrics, float(threshold), stl_written


# ---- Main Processing Function (single, fixed definition) ----
def process_dicom_folder(folder_path, plotly_output, stl_output=None,
                         prefer_hu=True, window_center=None, window_width=None,
                         mask_method='otsu', manual_threshold=None,
                         min_size_fraction=0.0005, closing_iter=2, pad=2,
                         resample=False, new_spacing=(1.0, 1.0, 1.0)):
    """
    Process DICOM folder and export:
      - plotly_output: path to write Plotly HTML
      - stl_output: optional path to write STL
    Returns a result dict with keys expected by templates.
    """
    # Read & sort
    datasets = read_dicom_datasets(folder_path)
    datasets_sorted, spacing = get_spacing_and_sort(datasets)
    pz, py, px = spacing

    # Stack to volume (HU preferred)
    vol = stack_volume(datasets_sorted, prefer_hu=prefer_hu)

    # Normalize (for mask creation)
    vol_norm = window_normalize(vol, window_center, window_width)

    # Make mask
    mask, used_th = make_binary_mask(vol_norm, method=mask_method, manual_threshold=manual_threshold, pct=0.5)
    mask_clean = clean_mask(mask, min_size_fraction=min_size_fraction, closing_iter=closing_iter)

    # Crop
    vol_crop_norm, mask_crop, origin = crop_to_bbox(vol_norm, mask_clean, pad=pad)
    vol_crop_orig, _, _ = crop_to_bbox(vol, mask_clean, pad=pad)  # original HU values cropped

    if vol_crop_norm.size == 0 or mask_crop.sum() == 0:
        raise RuntimeError("No foreground found after cleaning — check threshold or input series")

    # Optional resampling to new spacing
    if resample:
        # compute scale factors from original spacing to new_spacing
        zf = spacing[0] / new_spacing[0] if new_spacing[0] > 0 else 1.0
        yf = spacing[1] / new_spacing[1] if new_spacing[1] > 0 else 1.0
        xf = spacing[2] / new_spacing[2] if new_spacing[2] > 0 else 1.0
        try:
            vol_for_mc = zoom(vol_crop_norm, (zf, yf, xf), order=1)
            final_spacing = new_spacing
        except Exception:
            vol_for_mc = vol_crop_norm
            final_spacing = spacing
    else:
        vol_for_mc = vol_crop_norm
        final_spacing = spacing

    # Smooth a bit to reduce noise for marching cubes
    try:
        vol_for_mc = ndi.gaussian_filter(vol_for_mc, sigma=0.5)
    except Exception:
        pass

    # Choose mesh threshold (in normalized units or HU depending on prefer_hu)
    mc_threshold = manual_threshold if manual_threshold is not None else float(np.mean(vol_for_mc))

    # Extract mesh and export Plotly HTML + optional STL
    mesh_metrics, used_mesh_threshold, stl_written = extract_mesh_and_export(vol_for_mc, final_spacing,
                                                                            plotly_output, stl_output, threshold=mc_threshold)

    # Compute voxel-based metrics using cropped mask (use original spacing)
    # If resampled, voxel_volume_mm3 should reflect final_spacing
    voxel_count = int(np.count_nonzero(mask_crop))
    voxel_volume_mm3 = float(final_spacing[0] * final_spacing[1] * final_spacing[2])
    estimated_volume_mm3 = float(voxel_count * voxel_volume_mm3)
    estimated_volume_cm3 = estimated_volume_mm3 / 1000.0

    # Bounding box in mm (based on vol_for_mc shape and final_spacing)
    cz, ch, cw = vol_for_mc.shape
    bbox_mm = {
        "depth_mm": float(cz * final_spacing[0]),
        "height_mm": float(ch * final_spacing[1]),
        "width_mm": float(cw * final_spacing[2])
    }

    # centroid: compute from mask_crop (in cropped voxel coordinates) and map to mm in original coordinate system
    coords = np.argwhere(mask_crop > 0)
    if coords.size:
        centroid_vox = coords.mean(axis=0)  # z,y,x within cropped mask
        centroid_mm = [
            float(origin[0] * final_spacing[0] + centroid_vox[0] * final_spacing[0]),
            float(origin[1] * final_spacing[1] + centroid_vox[1] * final_spacing[1]),
            float(origin[2] * final_spacing[2] + centroid_vox[2] * final_spacing[2])
        ]
    else:
        centroid_mm = [0.0, 0.0, 0.0]

    result = {
        "plotly_html": plotly_output,
        "stl": stl_written if stl_written else stl_output,
        "threshold_used": float(used_mesh_threshold if used_mesh_threshold is not None else used_th),
        "mesh_metrics": mesh_metrics,
        "voxel_count": voxel_count,
        "voxel_volume_mm3": voxel_volume_mm3,
        "estimated_volume_mm3": estimated_volume_mm3,
        "estimated_volume_cm3": estimated_volume_cm3,
        "bbox_mm": bbox_mm,
        "centroid_mm": centroid_mm,
        "spacing_used": final_spacing,
        "crop_origin_voxel": origin,
        "crop_shape_voxel": tuple(int(x) for x in vol_for_mc.shape)
    }

    return result
