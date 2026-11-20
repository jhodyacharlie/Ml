import numpy as np
import tensorflow as tf

from cnn import MODEL_PATH, UKURAN
from ringan import BOBOT_PATH, muat_bobot, prediksi_batch

model = tf.keras.models.load_model(MODEL_PATH)

bobot = {}
konv = 0
bn = 0
dense = 0

for layer in model.layers:
  nama = layer.__class__.__name__
  w = layer.get_weights()

  if nama == "Conv2D":
    bobot[f"conv{konv}_k"] = w[0].astype(np.float32)
    bobot[f"conv{konv}_b"] = w[1].astype(np.float32)
    konv += 1
  elif nama == "BatchNormalization":
    bobot[f"bn{bn}_g"] = w[0].astype(np.float32)
    bobot[f"bn{bn}_b"] = w[1].astype(np.float32)
    bobot[f"bn{bn}_m"] = w[2].astype(np.float32)
    bobot[f"bn{bn}_v"] = w[3].astype(np.float32)
    bobot["eps"] = np.float32(layer.epsilon)
    bn += 1
  elif nama == "Dense":
    bobot[f"d{dense}_k"] = w[0].astype(np.float32)
    bobot[f"d{dense}_b"] = w[1].astype(np.float32)
    dense += 1

print("Conv2D:", konv, "BatchNorm:", bn, "Dense:", dense)

BOBOT_PATH.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(BOBOT_PATH, **bobot)
print("Bobot disimpan:", BOBOT_PATH)

rng = np.random.default_rng(0)
X = rng.integers(0, 256, (8, UKURAN, UKURAN, 1)).astype(np.uint8)

hasil_tf = model.predict(X, verbose=0)
hasil_np = prediksi_batch(muat_bobot(), X)

print("Selisih maksimum TensorFlow vs numpy:", float(np.abs(hasil_tf - hasil_np).max()))
print("Kelas teratas sama:", bool((hasil_tf.argmax(1) == hasil_np.argmax(1)).all()))
