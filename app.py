from fastapi import FastAPI, Request
from pydantic import BaseModel
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import torch

app = FastAPI()
model = MusicGen.get_pretrained('facebook/musicgen-small')
model.set_generation_params(duration=10)

class MusicPrompt(BaseModel):
    prompt: str
    duration: int = 10

@app.post("/generate")
def generate_music(data: MusicPrompt):
    model.set_generation_params(duration=data.duration)
    wav = model.generate([data.prompt])
    file_path = "/tmp/output.wav"
    audio_write(file_path, wav[0].cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
    return {"message": "Music generated", "file_path": file_path}
