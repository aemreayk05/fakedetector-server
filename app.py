import os
import time
import base64
import io
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import pipeline

# Flask app başlat
app = Flask(__name__)
CORS(app)

# Model pipeline (global)
model_pipeline = None

def load_model():
    global model_pipeline
    try:
        print("🔄 Model yükleniyor: haywoodsloan/ai-image-detector-deploy")
        model_pipeline = pipeline(
            "image-classification",
            model="haywoodsloan/ai-image-detector-deploy",
            device=0 if torch.cuda.is_available() else -1
        )
        print("✅ Model yüklendi.")
        return True
    except Exception as e:
        print(f"❌ Model yüklenemedi: {e}")
        return False

def preprocess_image(base64_string):
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))

        if image.mode != 'RGB':
            image = image.convert('RGB')

        max_size = 512
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        return image
    except Exception as e:
        print(f"❌ Görsel işleme hatası: {e}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_pipeline is not None,
        'timestamp': time.time()
    })

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if model_pipeline is None:
        return jsonify({'error': 'Model yüklenmedi', 'success': False}), 500

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'image alanı eksik', 'success': False}), 400

    base64_image = data['image']
    if ',' in base64_image:
        base64_image = base64_image.split(',')[1]

    image = preprocess_image(base64_image)
    if image is None:
        return jsonify({'error': 'Görsel işlenemedi', 'success': False}), 400

    try:
        start_time = time.time()
        results = model_pipeline(image)
        duration = time.time() - start_time

        best = max(results, key=lambda x: x['score'])
        label = best['label'].upper()
        confidence = float(best['score'])

        if 'FAKE' in label or 'AI' in label or 'GENERATED' in label:
            prediction_tr, prediction_en, is_fake = 'Sahte', 'Fake', True
        elif 'REAL' in label or 'AUTHENTIC' in label:
            prediction_tr, prediction_en, is_fake = 'Gerçek', 'Real', False
        else:
            is_fake = confidence < 0.5
            prediction_tr = 'Sahte' if is_fake else 'Gerçek'
            prediction_en = 'Fake' if is_fake else 'Real'

        return jsonify({
            'success': True,
            'prediction': prediction_tr,
            'prediction_en': prediction_en,
            'confidence': round(confidence * 100, 2),
            'processing_time': round(duration, 2),
            'all_results': [
                {'label': r['label'], 'score': round(r['score'] * 100, 2)}
                for r in results
            ],
            'timestamp': time.time()
        })

    except Exception as e:
        print(f"❌ Tahmin hatası: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

# Render bu dosyayı doğrudan çalıştırmaz, gunicorn başlatır.
# Ancak localde test etmek istersen:
if __name__ == '__main__':
    if load_model():
        app.run(host='0.0.0.0', port=5000)
