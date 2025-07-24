from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
from PIL import Image
import io
import base64
import time

app = Flask(__name__)
CORS(app)

# Hugging Face model URL ve API token
HF_API_URL = "https://api-inference.huggingface.co/models/haywoodsloan/ai-image-detector-deploy"
HF_TOKEN = os.getenv("HF_TOKEN")  # Environment variable olarak ayarlanmalı

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

@app.route("/")
def home():
    return jsonify({
        "message": "FakeDetector API Server",
        "model": "haywoodsloan/ai-image-detector-deploy",
        "endpoints": {
            "health": "/health",
            "model_info": "/model-info", 
            "analyze": "/analyze"
        }
    })

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "model": "haywoodsloan/ai-image-detector-deploy",
        "model_loaded": True,
        "timestamp": time.time()
    })

@app.route("/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "model_name": "haywoodsloan/ai-image-detector-deploy",
        "model_type": "SwinV2 (Swin Transformer V2)",
        "author": "haywoodsloan",
        "size": "781 MB",
        "description": "AI vs Real image classification model",
        "url": "https://huggingface.co/haywoodsloan/ai-image-detector-deploy"
    })

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        # Request kontrolü
        if 'image' not in request.json:
            return jsonify({"error": "Görsel bulunamadı"}), 400

        # Base64 görseli decode et
        image_data = request.json['image']
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)

        # Hugging Face'e gönder
        response = requests.post(HF_API_URL, headers=headers, data=image_bytes)

        if response.status_code != 200:
            return jsonify({"error": f"Hugging Face API hatası: {response.status_code}", "detail": response.text}), 500

        result = response.json()
        
        # Sonuçları parse et
        if len(result) >= 2:
            result1 = result[0]
            result2 = result[1]
            
            if 'FAKE' in result1['label'].upper():
                fake_prob = result1['score'] * 100
                real_prob = result2['score'] * 100
            else:
                real_prob = result1['score'] * 100
                fake_prob = result2['score'] * 100
            
            prediction = "Gerçek" if real_prob > fake_prob else "Sahte"
            confidence = max(real_prob, fake_prob)
        else:
            prediction = "Bilinmiyor"
            confidence = 0
            real_prob = 0
            fake_prob = 0

        # Sonucu döndür
        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "probabilities": {
                "real": round(real_prob, 2),
                "fake": round(fake_prob, 2)
            },
            "model_used": "haywoodsloan/ai-image-detector-deploy",
            "model_info": "SwinV2-based AI vs Real detection"
        })

    except Exception as e:
        return jsonify({"error": "Sunucu hatası", "message": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
