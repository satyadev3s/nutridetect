import json
import os

import numpy as np
import tensorflow as tf


def load_model_with_fallback(candidates):
    for p in candidates:
        if os.path.exists(p):
            return tf.keras.models.load_model(p), p
    raise FileNotFoundError(f'Model not found. Checked: {candidates}')


def preprocess(path):
    img = tf.keras.utils.load_img(path, target_size=(224, 224))
    arr = tf.keras.utils.img_to_array(img)
    return np.expand_dims(arr, axis=0) / 255.0


model, model_path = load_model_with_fallback(['human_model.keras', 'human_model.h5'])
print('Loaded model:', model_path)

img_path = 'dataset/test/1/7.png'
arr = preprocess(img_path)

prediction = float(model.predict(arr, verbose=0)[0][0])
print(f'Raw probability (class 1): {prediction:.4f}')

if prediction >= 0.5:
    print('Human')
else:
    print('Not Human')
