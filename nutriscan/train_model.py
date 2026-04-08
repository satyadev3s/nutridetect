from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
SEED = 42
DATASET_DIR = Path("dataset")
if not DATASET_DIR.exists():
    sibling_dataset_dir = Path(__file__).resolve().parents[1] / "dataset"
    if sibling_dataset_dir.exists():
        DATASET_DIR = sibling_dataset_dir
MODEL_OUTPUT = Path("malnutrition_model.keras")
CLASS_NAMES = ("malnourished", "normal")


def validate_dataset_layout(dataset_dir: Path) -> None:
    missing = [class_name for class_name in CLASS_NAMES if not (dataset_dir / class_name).is_dir()]
    if missing:
        raise FileNotFoundError(
            "Dataset folders not found. Create:\n"
            f"  {dataset_dir / 'malnourished'}\n"
            f"  {dataset_dir / 'normal'}\n"
            "and place training images inside them."
        )


def build_datasets(dataset_dir: Path):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        labels="inferred",
        label_mode="binary",
        class_names=list(CLASS_NAMES),
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        labels="inferred",
        label_mode="binary",
        class_names=list(CLASS_NAMES),
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.shuffle(500, seed=SEED).prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    return train_ds, val_ds


def build_model():
    augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )

    base_model = keras.applications.MobileNetV2(
        input_shape=IMAGE_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = keras.Input(shape=IMAGE_SIZE + (3,))
    x = augmentation(inputs)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def main():
    validate_dataset_layout(DATASET_DIR)
    train_ds, val_ds = build_datasets(DATASET_DIR)
    model = build_model()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_OUTPUT),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    results = model.evaluate(val_ds, verbose=0)
    metrics = dict(zip(model.metrics_names, results))
    print("Validation metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    model.save(MODEL_OUTPUT)
    print(f"Saved trained model to {MODEL_OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
