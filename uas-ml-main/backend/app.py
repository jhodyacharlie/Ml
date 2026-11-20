import json
import os

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from ringan import LABEL_PATH, UKURAN, bersihkan, muat_bobot, prediksi_batch

ORIGINS = [
  o.strip()
  for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
  if o.strip()
]

app = FastAPI()

app.add_middleware(
  CORSMiddleware,
  allow_origins=ORIGINS,
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

bobot = muat_bobot()

with open(LABEL_PATH, "r", encoding="utf-8") as f:
  kelas = json.load(f)["kelas"]


@app.get("/")
@app.get("/api")
def home():
  return {"message": "Backend klasifikasi CNN aktif"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
  contents = await file.read()

  img_array = np.frombuffer(contents, np.uint8)
  img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

  if img is None:
    raise HTTPException(
      status_code=400,
      detail="Gambar tidak valid"
    )

  img_bersih = bersihkan(img)

  if img_bersih is None:
    raise HTTPException(
      status_code=400,
      detail="Tulisan tidak berhasil dideteksi"
    )

  img_resize = cv2.resize(
    img_bersih,
    (UKURAN, UKURAN),
    interpolation=cv2.INTER_AREA
  )

  X = img_resize[np.newaxis, ..., np.newaxis]

  probabilitas = prediksi_batch(bobot, X)[0]

  urutan = np.argsort(probabilitas)[::-1][:3]

  hasil = [
    {
      "penulis": kelas[int(i)],
      "confidence": float(probabilitas[i] * 100)
    }
    for i in urutan
  ]

  return {
    "prediksi": hasil[0]["penulis"],
    "confidence": hasil[0]["confidence"],
    "top_3": hasil
  }
