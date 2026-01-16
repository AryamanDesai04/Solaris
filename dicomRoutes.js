// dicomRoutes.js
import express from "express";
import axios from "axios";
import fs from "fs";
import path from "path";
import { GoogleAuth } from "google-auth-library";

const router = express.Router();

/* ====== CONFIG ====== */
const PROJECT_ID = "your-project-id";
const LOCATION = "asia-south1"; // India region preferred
const DATASET_ID = "medisphere_dataset";
const DICOM_STORE_ID = "medisphere_dicom_store";

const BASE_URL = `https://healthcare.googleapis.com/v1/projects/${PROJECT_ID}/locations/${LOCATION}/datasets/${DATASET_ID}/dicomStores/${DICOM_STORE_ID}/dicomWeb`;

/* ====== AUTH ====== */
const auth = new GoogleAuth({
  keyFile: "service-account.json",
  scopes: ["https://www.googleapis.com/auth/cloud-platform"],
});

/* ====== UPLOAD DICOM ====== */
router.post("/upload", async (req, res) => {
  try {
    const { dicomBase64 } = req.body;
    const dicomBuffer = Buffer.from(dicomBase64, "base64");

    const client = await auth.getClient();
    const token = await client.getAccessToken();

    const response = await axios.post(
      `${BASE_URL}/studies`,
      dicomBuffer,
      {
        headers: {
          "Authorization": `Bearer ${token.token}`,
          "Content-Type": "application/dicom",
        },
      }
    );

    res.status(200).json({
      message: "DICOM uploaded successfully",
      response: response.data,
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/* ====== LIST STUDIES ====== */
router.get("/studies", async (req, res) => {
  try {
    const client = await auth.getClient();
    const token = await client.getAccessToken();

    const response = await axios.get(
      `${BASE_URL}/studies`,
      {
        headers: {
          "Authorization": `Bearer ${token.token}`,
        },
      }
    );

    res.status(200).json(response.data);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/* ====== RETRIEVE SERIES ====== */
router.get("/studies/:studyId/series", async (req, res) => {
  try {
    const { studyId } = req.params;

    const client = await auth.getClient();
    const token = await client.getAccessToken();

    const response = await axios.get(
      `${BASE_URL}/studies/${studyId}/series`,
      {
        headers: {
          "Authorization": `Bearer ${token.token}`,
        },
      }
    );

    res.status(200).json(response.data);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

export default router;
