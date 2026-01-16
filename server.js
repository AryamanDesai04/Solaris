// server.js
import express from "express";
import dicomRoutes from "./dicomRoutes.js";
import bodyParser from "body-parser";

const app = express();
app.use(bodyParser.json({ limit: "50mb" }));

app.use("/api/dicom", dicomRoutes);

const PORT = 5000;
app.listen(PORT, () => {
  console.log(`MediSphere backend running on port ${PORT}`);
});
