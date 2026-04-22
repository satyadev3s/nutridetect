import json
import os
from pathlib import Path

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


def list_images(folder):
    exts = {'.png', '.jpg', '.jpeg', '.webp'}
    return [
        str(p)
        for p in Path(folder).iterdir()
        if p.is_file() and p.suffix.lower() in exts
    ]


def predict_single(model, img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0
    return float(model.predict(arr, verbose=0)[0][0])


def detect_face_count(img_path):
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    frame = cv2.imread(img_path)
    if frame is None:
        return 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces)


def confusion(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn


def metrics(tp, tn, fp, fn):
    total = tp + tn + fp + fn
    acc = (tp + tn) / (total + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    specificity = tn / (tn + fp + 1e-9)
    bal_acc = (recall + specificity) / 2.0
    return {
        'accuracy': round(acc, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'specificity': round(specificity, 4),
        'balanced_accuracy': round(bal_acc, 4),
    }


def main():
    thresholds = load_json(
        'model_thresholds.json',
        {
            'human_threshold': 0.5,
            'face_fallback_threshold': 0.15,
            'malnutrition_threshold': 0.5,
            'malnutrition_uncertain_margin': 0.12,
        },
    )

    human_model, human_path = load_model_with_fallback(['human_model.keras', 'human_model.h5'])
    mal_model, mal_path = load_model_with_fallback(['malnutrition_model.keras', 'malnutrition_model.h5'])
    mal_labels_map = load_json('malnutrition_labels.json', {}).get('class_indices', {})
    mal_idx = mal_labels_map.get('malnutrition', 0)

    # Human evaluation with face fallback logic
    human_0 = list_images('human_dataset/test/0')
    human_1 = list_images('human_dataset/test/1')
    human_paths = human_0 + human_1
    y_true_h = np.array([0] * len(human_0) + [1] * len(human_1), dtype=np.int32)

    y_pred_h = []
    for p in human_paths:
        prob = predict_single(human_model, p)
        faces = detect_face_count(p)
        is_human = prob >= float(thresholds['human_threshold']) or (
            faces > 0 and prob >= float(thresholds['face_fallback_threshold'])
        )
        y_pred_h.append(1 if is_human else 0)
    y_pred_h = np.array(y_pred_h, dtype=np.int32)

    h_tp, h_tn, h_fp, h_fn = confusion(y_true_h, y_pred_h)
    human_metrics = metrics(h_tp, h_tn, h_fp, h_fn)

    # Malnutrition evaluation
    mal_m = list_images('malnutrition_dataset/test/malnutrition')
    mal_n = list_images('malnutrition_dataset/test/normal')
    mal_paths = mal_m + mal_n
    y_true_m = np.array([1] * len(mal_m) + [0] * len(mal_n), dtype=np.int32)

    y_pred_m = []
    uncertain_count = 0
    for p in mal_paths:
        raw = predict_single(mal_model, p)
        mal_prob = raw if mal_idx == 1 else 1.0 - raw
        delta = abs(mal_prob - float(thresholds['malnutrition_threshold']))
        if delta < float(thresholds['malnutrition_uncertain_margin']):
            uncertain_count += 1
        y_pred_m.append(1 if mal_prob >= float(thresholds['malnutrition_threshold']) else 0)
    y_pred_m = np.array(y_pred_m, dtype=np.int32)

    m_tp, m_tn, m_fp, m_fn = confusion(y_true_m, y_pred_m)
    mal_metrics = metrics(m_tp, m_tn, m_fp, m_fn)

    report = {
        'models': {
            'human_model': human_path,
            'malnutrition_model': mal_path,
        },
        'thresholds_used': thresholds,
        'human_detection': {
            'confusion_matrix': {
                'tp': h_tp,
                'tn': h_tn,
                'fp': h_fp,
                'fn': h_fn,
            },
            'metrics': human_metrics,
            'samples': int(len(y_true_h)),
        },
        'malnutrition_detection': {
            'confusion_matrix': {
                'tp': m_tp,
                'tn': m_tn,
                'fp': m_fp,
                'fn': m_fn,
            },
            'metrics': mal_metrics,
            'samples': int(len(y_true_m)),
            'uncertain_if_enabled_count': int(uncertain_count),
        },
    }

    with open('evaluation_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print('Saved evaluation_report.json')
    print('Human confusion:', report['human_detection']['confusion_matrix'])
    print('Human metrics:', human_metrics)
    print('Malnutrition confusion:', report['malnutrition_detection']['confusion_matrix'])
    print('Malnutrition metrics:', mal_metrics)
    print('Malnutrition uncertain count:', uncertain_count)


if __name__ == '__main__':
    main()
