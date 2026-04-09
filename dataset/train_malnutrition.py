import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print('Starting malnutrition model training...')

# ✅ Train generator (with rescale + augmentation)
train_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)

# ✅ Test generator
test_gen = ImageDataGenerator(rescale=1.0 / 255)

# Load data
train_data = train_gen.flow_from_directory(
    'malnutrition_dataset/train',
    target_size=(224, 224),
    class_mode='binary',
    batch_size=16,
)

test_data = test_gen.flow_from_directory(
    'malnutrition_dataset/test',
    target_size=(224, 224),
    class_mode='binary',
    batch_size=16,
    shuffle=False,
)

print('Class indices:', train_data.class_indices)

# Save labels
with open('malnutrition_labels.json', 'w', encoding='utf-8') as f:
    json.dump({'class_indices': train_data.class_indices}, f, indent=2)

# ✅ MobileNet model
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid'),
])

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('malnutrition_model.keras', monitor='val_accuracy', save_best_only=True),
]

# Train
model.fit(train_data, epochs=15, validation_data=test_data, callbacks=callbacks)

# Evaluate
loss, acc = model.evaluate(test_data, verbose=0)
print(f'Test Accuracy: {acc:.4f}')

# Save model
model.save('malnutrition_model.h5')

print('Training complete ✅')