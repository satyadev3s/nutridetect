import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf


def load_model_with_fallback(candidates):
    for p in candidates:
        if os.path.exists(p):
            return tf.keras.models.load_model(p), p
    raise FileNotFoundError(f'Model not found. Checked: {candidates}')


def load_class_indices(path):
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f).get('class_indices', {})


def list_images(folder):
    exts = {'.png', '.jpg', '.jpeg', '.webp'}
    return [
        str(p)
        for p in Path(folder).iterdir()
        if p.is_file() and p.suffix.lower() in exts
    ]


def predict_batch(model, paths, batch_size=32):
    probs = []
    for i in range(0, len(paths), batch_size):
        chunk = paths[i:i + batch_size]
        arr = []
        for p in chunk:
            img = tf.keras.utils.load_img(p, target_size=(224, 224))
            x = tf.keras.utils.img_to_array(img) / 255.0
            arr.append(x)
        arr = np.array(arr)
        out = model.predict(arr, verbose=0).reshape(-1)
        probs.extend(out.tolist())
    return np.array(probs, dtype=np.float32)


def best_threshold(y_true, probs, min_t=0.05, max_t=0.95, step=0.01):
    best = None
    for t in np.arange(min_t, max_t + 1e-9, step):
        y_pred = (probs >= t).astype(np.int32)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tpr = tp / (tp + fn + 1e-9)
        tnr = tn / (tn + fp + 1e-9)
        bal_acc = (tpr + tnr) / 2.0
        acc = (tp + tn) / (tp + tn + fp + fn + 1e-9)
        row = (bal_acc, acc, float(t), tp, tn, fp, fn)
        if best is None or row[0] > best[0]:
            best = row
    return best


def main():
    human_model, human_path = load_model_with_fallback(['human_model.keras', 'human_model.h5'])
    mal_model, mal_path = load_model_with_fallback(['malnutrition_model.keras', 'malnutrition_model.h5'])
    mal_class_indices = load_class_indices('malnutrition_labels.json')

    # Human threshold calibration
    human_paths = list_images('dataset/test/0') + list_images('dataset/test/1')
    human_labels = np.array([0] * len(list_images('dataset/test/0')) + [1] * len(list_images('dataset/test/1')), dtype=np.int32)
    human_probs = predict_batch(human_model, human_paths)

    h_bal_acc, h_acc, h_t, h_tp, h_tn, h_fp, h_fn = best_threshold(human_labels, human_probs)

    # Malnutrition threshold calibration (convert to P(malnutrition))
    mal_test_m = list_images('malnutrition_dataset/test/malnutrition')
    mal_test_n = list_images('malnutrition_dataset/test/normal')
    mal_paths = mal_test_m + mal_test_n
    mal_labels = np.array([1] * len(mal_test_m) + [0] * len(mal_test_n), dtype=np.int32)
    mal_raw_probs = predict_batch(mal_model, mal_paths)

    malnutrition_index = mal_class_indices.get('malnutrition', 0)
    if malnutrition_index == 1:
        mal_probs = mal_raw_probs
    else:
        mal_probs = 1.0 - mal_raw_probs

    m_bal_acc, m_acc, m_t, m_tp, m_tn, m_fp, m_fn = best_threshold(mal_labels, mal_probs)

    payload = {
        'human_threshold': round(h_t, 4),
        'face_fallback_threshold': 0.15,
        'malnutrition_threshold': round(m_t, 4),
        'malnutrition_uncertain_margin': 0.12,
        'calibration_summary': {
            'human': {
                'model_path': human_path,
                'balanced_accuracy': round(h_bal_acc, 4),
                'accuracy': round(h_acc, 4),
                'tp': h_tp,
                'tn': h_tn,
                'fp': h_fp,
                'fn': h_fn,
                'samples': int(len(human_labels)),
            },
            'malnutrition': {
                'model_path': mal_path,
                'balanced_accuracy': round(m_bal_acc, 4),
                'accuracy': round(m_acc, 4),
                'tp': m_tp,
                'tn': m_tn,
                'fp': m_fp,
                'fn': m_fn,
                'samples': int(len(mal_labels)),
            },
        },
    }

    with open('model_thresholds.json', 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    print('Saved thresholds to model_thresholds.json')
    print('Human threshold:', payload['human_threshold'])
    print('Malnutrition threshold:', payload['malnutrition_threshold'])


if __name__ == '__main__':
    main()
