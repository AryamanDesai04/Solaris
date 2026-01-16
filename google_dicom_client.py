# google_dicom_client.py
import requests
import google.auth
from google.auth.transport.requests import AuthorizedSession

PROJECT_ID = "your-project-id"
LOCATION = "us-central1"
DATASET_ID = "medisphere_dataset"
DICOM_STORE_ID = "medisphere_dicom_store"

DICOM_STORE_URL = (
    f"https://healthcare.googleapis.com/v1/projects/{PROJECT_ID}"
    f"/locations/{LOCATION}/datasets/{DATASET_ID}"
    f"/dicomStores/{DICOM_STORE_ID}/dicomWeb/studies"
)

def upload_dicom_to_google(dicom_file_path):
    """
    Upload a single DICOM file to Google Cloud DICOM Store
    """
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    authed_session = AuthorizedSession(credentials)

    headers = {
        "Content-Type": "application/dicom"
    }

    with open(dicom_file_path, "rb") as f:
        response = authed_session.post(
            DICOM_STORE_URL,
            headers=headers,
            data=f
        )

    if response.status_code not in (200, 201):
        raise RuntimeError(
            f"Google DICOM upload failed: {response.status_code} {response.text}"
        )

    return True
