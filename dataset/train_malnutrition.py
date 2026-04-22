import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / 'malnutrition_dataset'
TRAIN_DIR = DATASET_DIR / 'train'
TEST_DIR = DATASET_DIR / 'test'
MODEL_H5_PATH = BASE_DIR / 'malnutrition_model.h5'
MODEL_KERAS_PATH = BASE_DIR / 'malnutrition_model.keras'
LABELS_PATH = BASE_DIR / 'malnutrition_labels.json'


def count_images(folder):
    exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)


def compute_class_weights(class_counts, class_indices):
    total = sum(class_counts.values())
    n_classes = max(1, len(class_indices))
    weights = {}
    for class_name, class_index in class_indices.items():
        count = class_counts.get(class_name, 1)
        weights[class_index] = total / (n_classes * max(1, count))
    return weights


def main():
    print('Starting malnutrition model training...')
    print(f'Using dataset: {DATASET_DIR}')

    for required in [TRAIN_DIR / 'malnutrition', TRAIN_DIR / 'normal', TEST_DIR / 'malnutrition', TEST_DIR / 'normal']:
        if not required.exists():
            raise FileNotFoundError(f'Missing required folder: {required}')

    train_counts = {
        'malnutrition': count_images(TRAIN_DIR / 'malnutrition'),
        'normal': count_images(TRAIN_DIR / 'normal'),
    }
    test_counts = {
        'malnutrition': count_images(TEST_DIR / 'malnutrition'),
        'normal': count_images(TEST_DIR / 'normal'),
    }
    print('Train counts:', train_counts)
    print('Test counts:', test_counts)

    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    test_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_data = train_gen.flow_from_directory(
        str(TRAIN_DIR),
        target_size=(224, 224),
        class_mode='binary',
        batch_size=16,
    )
    test_data = test_gen.flow_from_directory(
        str(TEST_DIR),
        target_size=(224, 224),
        class_mode='binary',
        batch_size=16,
        shuffle=False,
    )

    print('Class indices:', train_data.class_indices)

    with LABELS_PATH.open('w', encoding='utf-8') as f:
        json.dump({'class_indices': train_data.class_indices}, f, indent=2)

    class_weights = compute_class_weights(train_counts, train_data.class_indices)
    print('Class weights:', class_weights)

    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(MODEL_KERAS_PATH), monitor='val_accuracy', save_best_only=True),
    ]

    model.fit(
        train_data,
        epochs=15,
        validation_data=test_data,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    _, acc = model.evaluate(test_data, verbose=0)
    print(f'Test Accuracy: {acc:.4f}')

    model.save(str(MODEL_H5_PATH))
    print(f'Saved: {MODEL_H5_PATH}')
    print(f'Saved: {MODEL_KERAS_PATH}')
    print(f'Saved labels: {LABELS_PATH}')
    print('Training complete')


if __name__ == '__main__':
    main()
