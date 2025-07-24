from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
from PIL import Image
import io
import base64

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
    return "Fake Detector API is running!"

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "Görsel dosyası eksik"}), 400

        image_file = request.files['image']
        image_bytes = image_file.read()

        # Hugging Face'e gönder
        response = requests.post(HF_API_URL, headers=headers, data=image_bytes)

        if response.status_code != 200:
            return jsonify({"error": f"Hugging Face API hatası: {response.status_code}", "detail": response.text}), 500

        result = response.json()
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": "Sunucu hatası", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
