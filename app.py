from flask import Flask, request, jsonify, send_file
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import torch
import os
import uuid

app = Flask(__name__)
model = MusicGen.get_pretrained('small')
model.set_generation_params(duration=30)

@app.route("/generate", methods=["POST"])
def generate_music():
    data = request.json
    prompt = data.get("prompt", "ambient cinematic background music")
    duration = int(data.get("duration", 30))
    uid = str(uuid.uuid4())
    filename = f"output/{uid}.wav"

    os.makedirs("output", exist_ok=True)
    model.set_generation_params(duration=duration)
    wav = model.generate([prompt])
    audio_write(filename.replace(".wav", ""), wav[0].cpu(), sampling_rate=32000)

    url = request.host_url + f"download/{uid}.wav"
    return jsonify({"url": url, "status": "success"})

@app.route("/download/<filename>")
def download_file(filename):
    filepath = f"output/{filename}"
    return send_file(filepath, mimetype="audio/wav")

@app.route("/")
def home():
    return "MusicGen API is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
