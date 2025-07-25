from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
from PIL import Image
import io
import base64
import time

app = Flask(__name__)  # ← BU SATIR EKSİK SENDE
CORS(app)

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if 'image' not in request.json:
            return jsonify({"error": "Görsel bulunamadı"}), 400

        image_data = request.json['image']
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)

        # ✅ DÜZELTİLEN KISIM: multipart/form-data olarak gönderiyoruz
        files = {"file": ("image.png", image_bytes, "image/png")}
        response = requests.post(HF_API_URL, headers=headers, files=files)

        if response.status_code != 200:
            return jsonify({
                "error": f"Hugging Face API hatası: {response.status_code}",
                "detail": response.text
            }), 500

        result = response.json()
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
        import traceback
        traceback.print_exc()  # sunucu logunda hatayı göster
        return jsonify({"error": "Sunucu hatası", "message": str(e)}), 500
