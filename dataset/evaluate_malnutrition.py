import json
from pathlib import Path

import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / 'malnutrition_dataset'
REPORT_PATH = BASE_DIR / 'malnutrition_evaluation_report.json'
LABELS_PATH = BASE_DIR / 'malnutrition_labels.json'


def load_model_with_fallback(candidates):
    errors = []
    for relative in candidates:
        path = BASE_DIR / relative
        if path.exists():
            try:
                return tf.keras.models.load_model(str(path)), path
            except Exception as exc:
                errors.append(f'{path}: {exc}')
    if errors:
        raise RuntimeError('Model loading failed:\n' + '\n'.join(errors))
    raise FileNotFoundError(f'Model not found. Checked: {candidates}')


def list_images(folder):
    exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    return [str(p) for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]


def load_labels():
    if not LABELS_PATH.exists():
        return {}
    with LABELS_PATH.open('r', encoding='utf-8') as f:
        return json.load(f).get('class_indices', {})


def predict_mal_prob(model, img_path, mal_idx):
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0
    raw = float(model.predict(arr, verbose=0)[0][0])
    return (1.0 - raw) if mal_idx == 0 else raw


def confusion(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn


def metric_dict(tp, tn, fp, fn):
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / max(1, total)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    balanced_accuracy = (recall + specificity) / 2.0
    return {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'specificity': round(specificity, 4),
        'balanced_accuracy': round(balanced_accuracy, 4),
    }


def main():
    model, model_path = load_model_with_fallback(['malnutrition_model.h5', 'malnutrition_model.keras'])
    labels = load_labels()
    mal_idx = labels.get('malnutrition', 0)
    threshold = 0.5

    test_m = list_images(DATASET_DIR / 'test' / 'malnutrition')
    test_n = list_images(DATASET_DIR / 'test' / 'normal')
    paths = test_m + test_n
    y_true = np.array([1] * len(test_m) + [0] * len(test_n), dtype=np.int32)
    y_pred = np.array(
        [1 if predict_mal_prob(model, p, mal_idx) >= threshold else 0 for p in paths],
        dtype=np.int32,
    )

    tp, tn, fp, fn = confusion(y_true, y_pred)
    metrics = metric_dict(tp, tn, fp, fn)

    report = {
        'model_path': str(model_path),
        'dataset_test_path': str(DATASET_DIR / 'test'),
        'class_indices': labels,
        'threshold': threshold,
        'samples': {
            'total': int(len(y_true)),
            'malnutrition': int(len(test_m)),
            'normal': int(len(test_n)),
        },
        'confusion_matrix': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn},
        'metrics': metrics,
    }

    with REPORT_PATH.open('w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f'Model: {model_path}')
    print(f'Test samples: total={len(y_true)} malnutrition={len(test_m)} normal={len(test_n)}')
    print(f"Confusion: tp={tp} tn={tn} fp={fp} fn={fn}")
    print('Metrics:', metrics)
    print(f'Saved report: {REPORT_PATH}')


if __name__ == '__main__':
    main()
