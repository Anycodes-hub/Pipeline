from flask import Flask, request, jsonify, send_file
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import os
import uuid

app = Flask(__name__)
model = MusicGen.get_pretrained('small')  # 'small' model fits free servers
model.set_generation_params(duration=30)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', 'cinematic inspirational background music')
    duration = int(data.get('duration', 30))

    model.set_generation_params(duration=duration)

    uid = str(uuid.uuid4())
    filename = f"output/{uid}.wav"

    os.makedirs("output", exist_ok=True)

    wav = model.generate([prompt])
    audio_write(filename.replace(".wav", ""), wav[0].cpu(), sampling_rate=32000)

    # Return URL
    base_url = request.host_url.rstrip("/")
    return jsonify({"url": f"{base_url}/download/{uid}.wav", "status": "ok"})

@app.route('/download/<filename>')
def download(filename):
    return send_file(f"output/{filename}", mimetype='audio/wav')

@app.route('/')
def home():
    return '🎵 MusicGen API running.'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
