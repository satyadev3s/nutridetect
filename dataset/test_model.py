import os

import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_model_with_fallback(candidates):
    errors = []
    for p in candidates:
        abs_path = p if os.path.isabs(p) else os.path.join(BASE_DIR, p)
        if os.path.exists(abs_path):
            try:
                return tf.keras.models.load_model(abs_path), abs_path
            except Exception as exc:
                errors.append(f'{abs_path}: {exc}')
    if errors:
        raise RuntimeError('Model loading failed:\n' + '\n'.join(errors))
    raise FileNotFoundError(f'Model not found. Checked: {candidates}')


def preprocess(path):
    img = tf.keras.utils.load_img(path, target_size=(224, 224))
    arr = tf.keras.utils.img_to_array(img)
    return np.expand_dims(arr, axis=0) / 255.0


def pick_sample_image():
    exts = {'.png', '.jpg', '.jpeg', '.webp'}
    candidate_dirs = [
        os.path.join(BASE_DIR, 'human_dataset', 'test', '1'),
        os.path.join(BASE_DIR, 'human_dataset', 'test', '0'),
    ]
    for folder in candidate_dirs:
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if os.path.isfile(path) and os.path.splitext(name)[1].lower() in exts:
                return path
    raise FileNotFoundError('No sample image found under human_dataset/test/0 or human_dataset/test/1')


model, model_path = load_model_with_fallback(['human_model.h5', 'human_model.keras'])
print('Loaded model:', model_path)

img_path = pick_sample_image()
print('Using sample image:', img_path)
arr = preprocess(img_path)

prediction = float(model.predict(arr, verbose=0)[0][0])
print(f'Raw probability (class 1): {prediction:.4f}')

if prediction >= 0.5:
    print('Human')
else:
    print('Not Human')
