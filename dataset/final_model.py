import json
import os

import cv2
import numpy as np
import tensorflow as tf


def load_model_with_fallback(candidates):
    for p in candidates:
        if os.path.exists(p):
            return tf.keras.models.load_model(p), p
    raise FileNotFoundError(f'Model not found. Checked: {candidates}')


def load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_class_indices(path):
    data = load_json(path, {})
    return data.get('class_indices', {})


def preprocess(path):
    img = tf.keras.utils.load_img(path, target_size=(224, 224))
    arr = tf.keras.utils.img_to_array(img)
    return np.expand_dims(arr, axis=0) / 255.0


def detect_face_count(img_path):
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    frame = cv2.imread(img_path)
    if frame is None:
        return 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces)


def get_malnutrition_probability(mal_output_prob, class_indices):
    malnutrition_index = class_indices.get('malnutrition', 0)
    if malnutrition_index == 1:
        return mal_output_prob
    return 1.0 - mal_output_prob


DEFAULT_THRESHOLDS = {
    'human_threshold': 0.5,
    'face_fallback_threshold': 0.15,
    'malnutrition_threshold': 0.5,
    'malnutrition_uncertain_margin': 0.12,
}
thresholds = {**DEFAULT_THRESHOLDS, **load_json('model_thresholds.json', {})}

human_model, human_model_path = load_model_with_fallback(['human_model.keras', 'human_model.h5'])
mal_model, mal_model_path = load_model_with_fallback(['malnutrition_model.keras', 'malnutrition_model.h5'])
mal_class_indices = load_class_indices('malnutrition_labels.json')

print('Loaded human model:', human_model_path)
print('Loaded malnutrition model:', mal_model_path)
print('Malnutrition class indices:', mal_class_indices if mal_class_indices else 'default assumption')
print('Thresholds:', thresholds)

img_path = input('Enter image path: ').strip()
if not os.path.exists(img_path):
    raise FileNotFoundError(f'Image not found: {img_path}')

arr = preprocess(img_path)

human_prob = float(human_model.predict(arr, verbose=0)[0][0])
face_count = detect_face_count(img_path)
is_human = human_prob >= float(thresholds['human_threshold']) or (
    face_count > 0 and human_prob >= float(thresholds['face_fallback_threshold'])
)
print(f'Human probability (class 1): {human_prob:.4f}')
print(f'Faces detected: {face_count}')

if not is_human:
    print(f'Not human (confidence: {(1.0 - human_prob) * 100:.1f}%)')
    raise SystemExit(0)

mal_output = float(mal_model.predict(arr, verbose=0)[0][0])
malnutrition_prob = get_malnutrition_probability(mal_output, mal_class_indices)

delta = abs(malnutrition_prob - float(thresholds['malnutrition_threshold']))
if delta < float(thresholds['malnutrition_uncertain_margin']):
    print(
        'Uncertain nutrition result. Please upload a clear full-body image in good lighting. '
        f'(malnutrition probability: {malnutrition_prob * 100:.1f}%)'
    )
elif malnutrition_prob >= float(thresholds['malnutrition_threshold']):
    print(f'Malnutrition detected (confidence: {malnutrition_prob * 100:.1f}%)')
else:
    print(f'Healthy (confidence: {(1.0 - malnutrition_prob) * 100:.1f}%)')
