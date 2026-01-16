import os
import uuid
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from extract_engine import process_dicom_folder
from gif_generator import generate_gif_from_dicom
from utils import allowed_file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_ROOT = os.path.join(BASE_DIR, "uploads")
PLOTLY_ROOT = os.path.join(BASE_DIR, "static", "plotly")
GIF_ROOT = os.path.join(BASE_DIR, "static", "gifs")
MODEL_ROOT = os.path.join(BASE_DIR, "static", "models")

# Create necessary directories if they don't exist
for d in (UPLOAD_ROOT, PLOTLY_ROOT, GIF_ROOT, MODEL_ROOT):
    os.makedirs(d, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['UPLOAD_FOLDER'] = UPLOAD_ROOT

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'files[]' not in request.files:
        return render_template('error.html', message="No files part in request")

    files = request.files.getlist('files[]')
    valid_files = [f for f in files if f and allowed_file(f.filename)]
    if not valid_files:
        return render_template('error.html', message="No valid .dcm files uploaded")

    scan_id = str(uuid.uuid4())[:8]
    scan_folder = os.path.join(app.config['UPLOAD_FOLDER'], scan_id)
    os.makedirs(scan_folder, exist_ok=True)

    for f in valid_files:
        fname = secure_filename(f.filename)
        f.save(os.path.join(scan_folder, fname))

    gif_rel = os.path.join("gifs", f"{scan_id}.gif")
    gif_full = os.path.join(GIF_ROOT, f"{scan_id}.gif")
    try:
        generate_gif_from_dicom(scan_folder, gif_full)
    except Exception as e:
        return render_template('error.html', message=f"GIF generation failed: {e}")

    plotly_rel = os.path.join("plotly", f"{scan_id}.html")
    plotly_full = os.path.join(PLOTLY_ROOT, f"{scan_id}.html")
    stl_rel = os.path.join("models", f"{scan_id}.stl")
    stl_full = os.path.join(MODEL_ROOT, f"{scan_id}.stl")

    try:
        result = process_dicom_folder(
            scan_folder,
            plotly_output=plotly_full,
            stl_output=stl_full,
            prefer_hu=True,
            mask_method='otsu',
            min_size_fraction=0.0005,
            closing_iter=2,
            pad=2,
            resample=False
        )
    except Exception as e:
        return render_template('error.html', message=f"3D model generation failed: {e}")

    gif_path = "/" + os.path.join("static", gif_rel).replace("\\", "/")
    model_path = "/" + os.path.join("static", plotly_rel).replace("\\", "/")
    stl_link = "/" + os.path.join("static", stl_rel).replace("\\", "/")

    return render_template(
        'results.html',
        gif_path=gif_path,
        model_path=model_path,
        stl_link=stl_link if os.path.exists(stl_full) else None,
        metrics=result
    )

if __name__ == '__main__':
    app.run(debug=True)
