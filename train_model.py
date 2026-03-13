"""
train_model.py — Train a custom CNN on CIFAR-10 using TensorFlow/Keras
Run: python train_model.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Labels ──────────────────────────────────────────────────────────────────
CLASS_NAMES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# ── Load & preprocess CIFAR-10 ───────────────────────────────────────────────
def load_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test  = x_test.astype('float32')  / 255.0
    y_train = to_categorical(y_train, 10)
    y_test  = to_categorical(y_test,  10)
    print(f"Train: {x_train.shape}  Test: {x_test.shape}")
    return x_train, y_train, x_test, y_test

# ── Build CNN Model ──────────────────────────────────────────────────────────
def build_model():
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.40),

        # Classifier
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model

# ── Train ────────────────────────────────────────────────────────────────────
def train(epochs=50, batch_size=64):
    x_train, y_train, x_test, y_test = load_data()
    model = build_model()

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    datagen.fit(x_train)

    os.makedirs('saved_model', exist_ok=True)
    callbacks = [
        ModelCheckpoint('saved_model/best_model.h5', save_best_only=True, monitor='val_accuracy'),
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%  |  Test Loss: {test_loss:.4f}")

    plot_history(history)
    return model, history

# ── Plot Training History ────────────────────────────────────────────────────
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0a0f14')
    for ax in [ax1, ax2]:
        ax.set_facecolor('#0d1a1a')
        ax.tick_params(colors='#7a9a9a')
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e3030')

    ax1.plot(history.history['accuracy'],     color='#00ffc8', linewidth=2, label='Train Acc')
    ax1.plot(history.history['val_accuracy'], color='#7c83fd', linewidth=2, label='Val Acc', linestyle='--')
    ax1.set_title('Accuracy', color='#e0e8f0')
    ax1.legend(facecolor='#0d1a1a', labelcolor='#e0e8f0')

    ax2.plot(history.history['loss'],     color='#ff6b35', linewidth=2, label='Train Loss')
    ax2.plot(history.history['val_loss'], color='#7c83fd', linewidth=2, label='Val Loss', linestyle='--')
    ax2.set_title('Loss', color='#e0e8f0')
    ax2.legend(facecolor='#0d1a1a', labelcolor='#e0e8f0')

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("📊 Plot saved: training_history.png")
    plt.show()

if __name__ == '__main__':
    train(epochs=50, batch_size=64)
