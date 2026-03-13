"""
Image Recognition System - Flask Backend
Uses a MobileNetV2 model pretrained on ImageNet for classification.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import base64
import io
import time
import os

# Conditionally import heavy ML libs
try:
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
    from tensorflow.keras.preprocessing import image as keras_image
    from PIL import Image
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("TensorFlow not available - running in demo mode")

app = Flask(__name__)
CORS(app)

# Load model once at startup
model = None

def load_model():
    global model
    if ML_AVAILABLE and model is None:
        print("Loading MobileNetV2 model...")
        model = MobileNetV2(weights='imagenet')
        print("Model loaded successfully.")

def preprocess_image(img_bytes):
    """Convert raw image bytes to model-ready tensor."""
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def mock_predict(filename=""):
    """Return realistic mock predictions when ML libs unavailable."""
    categories = [
        [
            {"label": "tabby_cat", "display": "Tabby Cat", "confidence": 0.923},
            {"label": "persian_cat", "display": "Persian Cat", "confidence": 0.051},
            {"label": "tiger_cat", "display": "Tiger Cat", "confidence": 0.018},
            {"label": "lynx", "display": "Lynx", "confidence": 0.005},
            {"label": "cougar", "display": "Cougar", "confidence": 0.003},
        ],
        [
            {"label": "sports_car", "display": "Sports Car", "confidence": 0.871},
            {"label": "convertible", "display": "Convertible", "confidence": 0.072},
            {"label": "racer", "display": "Race Car", "confidence": 0.038},
            {"label": "minivan", "display": "Minivan", "confidence": 0.012},
            {"label": "pickup", "display": "Pickup Truck", "confidence": 0.007},
        ],
        [
            {"label": "golden_retriever", "display": "Golden Retriever", "confidence": 0.947},
            {"label": "labrador", "display": "Labrador", "confidence": 0.031},
            {"label": "kuvasz", "display": "Kuvasz", "confidence": 0.014},
            {"label": "clumber", "display": "Clumber Spaniel", "confidence": 0.005},
            {"label": "otterhound", "display": "Otterhound", "confidence": 0.003},
        ],
    ]
    import random
    chosen = random.choice(categories)
    return {
        "success": True,
        "mode": "demo",
        "predictions": chosen,
        "top_label": chosen[0]["display"],
        "top_confidence": chosen[0]["confidence"],
        "inference_time_ms": round(random.uniform(8, 25), 1),
        "model": "MobileNetV2 (Demo Mode)",
        "dataset": "ImageNet (1000 classes)"
    }


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "ml_available": ML_AVAILABLE,
        "model_loaded": model is not None,
        "mode": "full" if ML_AVAILABLE else "demo"
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    start_time = time.time()

    if 'file' not in request.files and 'image_b64' not in request.json:
        return jsonify({"error": "No image provided"}), 400

    try:
        # Get image bytes
        if 'file' in request.files:
            img_bytes = request.files['file'].read()
        else:
            data = request.json['image_b64']
            if ',' in data:
                data = data.split(',')[1]
            img_bytes = base64.b64decode(data)

        # Use real model or mock
        if ML_AVAILABLE:
            if model is None:
                load_model()
            img_tensor = preprocess_image(img_bytes)
            preds = model.predict(img_tensor)
            decoded = decode_predictions(preds, top=5)[0]
            predictions = [
                {
                    "label": p[1],
                    "display": p[1].replace('_', ' ').title(),
                    "confidence": float(p[2])
                }
                for p in decoded
            ]
            inference_ms = round((time.time() - start_time) * 1000, 1)
            return jsonify({
                "success": True,
                "mode": "full",
                "predictions": predictions,
                "top_label": predictions[0]["display"],
                "top_confidence": predictions[0]["confidence"],
                "inference_time_ms": inference_ms,
                "model": "MobileNetV2",
                "dataset": "ImageNet (1000 classes)"
            })
        else:
            return jsonify(mock_predict())

    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


@app.route('/api/model-info', methods=['GET'])
def model_info():
    return jsonify({
        "name": "MobileNetV2",
        "framework": "TensorFlow / Keras",
        "dataset": "ImageNet",
        "classes": 1000,
        "input_shape": [224, 224, 3],
        "parameters": "3.4M",
        "accuracy_top1": "71.8%",
        "accuracy_top5": "90.6%",
        "layers": [
            {"name": "Input Layer", "type": "InputLayer", "shape": "(224,224,3)"},
            {"name": "Conv2D + BN", "type": "Conv2D", "shape": "(112,112,32)"},
            {"name": "Depthwise Block x16", "type": "InvertedResidual", "shape": "(7,7,160)"},
            {"name": "Conv2D + BN", "type": "Conv2D", "shape": "(7,7,1280)"},
            {"name": "GlobalAvgPool", "type": "GlobalAveragePooling2D", "shape": "(1280,)"},
            {"name": "Dense + Softmax", "type": "Dense", "shape": "(1000,)"},
        ],
        "pipeline": [
            "Image Input",
            "Resize to 224×224",
            "Normalize [-1, 1]",
            "Feature Extraction (CNN)",
            "Global Average Pooling",
            "Softmax Classification",
            "Top-K Predictions"
        ]
    })


@app.route('/api/training-history', methods=['GET'])
def training_history():
    """Return simulated training history for charts."""
    epochs = list(range(1, 26))
    import math, random
    random.seed(42)

    def smooth(start, end, n, noise=0.01):
        return [
            round(start + (end - start) * (1 - math.exp(-4 * i / n)) + random.uniform(-noise, noise), 4)
            for i in range(1, n + 1)
        ]

    acc = smooth(0.45, 0.942, 25, 0.008)
    val_acc = smooth(0.40, 0.918, 25, 0.012)
    loss = smooth(0.85, 0.18, 25, 0.01)[::-1]
    val_loss = smooth(0.90, 0.22, 25, 0.015)[::-1]

    return jsonify({
        "epochs": epochs,
        "accuracy": acc,
        "val_accuracy": val_acc,
        "loss": loss,
        "val_loss": val_loss,
    })


if __name__ == '__main__':
    load_model()
    app.run(debug=True, port=5000)
