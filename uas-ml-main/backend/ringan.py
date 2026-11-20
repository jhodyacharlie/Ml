from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
UKURAN = 128
BOBOT_PATH = BASE_DIR / "models" / "bobot.npz"
LABEL_PATH = BASE_DIR / "models" / "kelas.json"


def bersihkan(img):
  if img is None:
    return None

  tinta = cv2.threshold(
    cv2.GaussianBlur(img, (3, 3), 0),
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
  )[1]

  h, w = tinta.shape

  k_h = cv2.getStructuringElement(
    cv2.MORPH_RECT, (min(max(40, w // 3), w), 1)
  )

  k_v = cv2.getStructuringElement(
    cv2.MORPH_RECT, (1, min(max(30, h // 2), h))
  )

  garis = (
    cv2.morphologyEx(tinta, cv2.MORPH_OPEN, k_h) | cv2.morphologyEx(tinta, cv2.MORPH_OPEN, k_v)
  )

  tinta = cv2.bitwise_and(
    tinta,
    cv2.bitwise_not(
      cv2.dilate(garis, np.ones((3, 3), np.uint8))
    )
  )

  ys, xs = np.where(tinta > 0)

  if len(ys) < 50:
    return None

  area = cv2.dilate(tinta, np.ones((5, 5), np.uint8)) > 0

  hasil = np.where(area, img, 255).astype(np.uint8)

  y1 = max(int(ys.min()) - 5, 0)
  y2 = min(int(ys.max()) + 6, h)
  x1 = max(int(xs.min()) - 5, 0)
  x2 = min(int(xs.max()) + 6, w)

  return hasil[y1:y2, x1:x2]


def muat_bobot(path=BOBOT_PATH):
  data = np.load(path)
  return {k: data[k] for k in data.files}


def konvolusi(x, kernel, bias):
  n, h, w, c = x.shape
  k = kernel.shape[0]
  pad = k // 2
  xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
  kolom = np.concatenate(
    [xp[:, i:i + h, j:j + w, :] for i in range(k) for j in range(k)],
    axis=-1
  )
  keluaran = kolom.reshape(n * h * w, k * k * c) @ kernel.reshape(k * k * c, -1) + bias
  return keluaran.reshape(n, h, w, -1)


def maxpool(x):
  n, h, w, c = x.shape
  x = x[:, :h // 2 * 2, :w // 2 * 2, :]
  return x.reshape(n, h // 2, 2, w // 2, 2, c).max(axis=(2, 4))


def prediksi_batch(bobot, x):
  x = x.astype(np.float32) / 255.0
  eps = float(bobot["eps"])

  for i in range(4):
    x = konvolusi(x, bobot[f"conv{i}_k"], bobot[f"conv{i}_b"])
    x = np.maximum(x, 0)
    x = (x - bobot[f"bn{i}_m"]) / np.sqrt(bobot[f"bn{i}_v"] + eps)
    x = x * bobot[f"bn{i}_g"] + bobot[f"bn{i}_b"]
    x = maxpool(x)

  x = x.reshape(x.shape[0], -1)
  x = np.maximum(x @ bobot["d0_k"] + bobot["d0_b"], 0)
  x = x @ bobot["d1_k"] + bobot["d1_b"]
  x = x - x.max(axis=1, keepdims=True)
  e = np.exp(x)
  return e / e.sum(axis=1, keepdims=True)
