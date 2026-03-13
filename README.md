# 🧠 Image Recognition System — Deep Learning

A complete end-to-end image classification system using MobileNetV2 + Flask backend + interactive HTML frontend.

---

## 📁 Project Structure

```
image-recognition/
├── backend/
│   ├── app.py              ← Flask REST API server
│   ├── train_model.py      ← Custom CNN training on CIFAR-10
│   └── requirements.txt    ← Python dependencies
└── frontend/
    └── index.html          ← Complete single-file frontend
```

---

## ⚙️ Setup & Run

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. (Optional) Train a Custom Model on CIFAR-10
```bash
python train_model.py
# Trains for up to 50 epochs with early stopping
# Saves best model to: saved_model/best_model.h5
# Saves training plot: training_history.png
```

### 3. Start the Backend
```bash
python app.py
# Server starts at http://localhost:5000
```

### 4. Open the Frontend
Open `frontend/index.html` directly in your browser.

> **Note:** If TensorFlow is not installed, the backend runs in **Demo Mode** with realistic mock predictions — no ML setup required!

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Check server & model status |
| POST | `/api/predict` | Classify an image (file upload or base64) |
| GET | `/api/model-info` | Model metadata & layer info |
| GET | `/api/training-history` | Simulated training curves |

### POST /api/predict
**Option A — File Upload:**
```bash
curl -X POST http://localhost:5000/api/predict \
  -F "file=@your_image.jpg"
```

**Option B — Base64 JSON:**
```json
POST /api/predict
{ "image_b64": "data:image/jpeg;base64,..." }
```

**Response:**
```json
{
  "success": true,
  "top_label": "Tabby Cat",
  "top_confidence": 0.923,
  "predictions": [
    { "label": "tabby", "display": "Tabby Cat", "confidence": 0.923 },
    ...
  ],
  "inference_time_ms": 12.4,
  "model": "MobileNetV2",
  "dataset": "ImageNet (1000 classes)"
}
```

---

## 🛠 Technologies

| Layer | Technology |
|-------|-----------|
| Language | Python 3.x |
| Deep Learning | TensorFlow 2.x / Keras |
| Model | MobileNetV2 (ImageNet) |
| Image Processing | OpenCV, Pillow |
| Backend | Flask + Flask-CORS |
| Frontend | HTML5, CSS3, Chart.js |
| Visualization | Matplotlib |

---

## 📊 Model Details

- **Architecture:** MobileNetV2 (depthwise separable convolutions)
- **Pretrained on:** ImageNet (1.2M images, 1000 classes)
- **Top-1 Accuracy:** 71.8% (ImageNet) / ~94% (CIFAR-10 fine-tuned)
- **Parameters:** 3.4M
- **Input Size:** 224×224×3
- **Inference Time:** ~12ms (CPU)
