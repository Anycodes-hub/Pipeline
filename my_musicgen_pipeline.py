from pipeline import Pipeline, Variable, pipe, entity
from pipeline.objects import File
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

@entity
class MusicgenModel:
    def __init__(self):
        self.model = None

    @pipe(on_startup=True, run_once=True)
    def load(self):
        self.model = MusicGen.get_pretrained("facebook/musicgen-small")

    @pipe
    def predict(self, prompt: str, duration: int = 15) -> File:
        self.model.set_generation_params(duration=duration)
        wavs = self.model.generate([prompt])
        file_path = "/tmp/output.wav"
        audio_write(file_path, wavs[0].cpu(), self.model.sample_rate, strategy="loudness", loudness_compressor=True)
        return File(path=file_path, allow_out_of_context_creation=True)

with Pipeline() as p:
    prompt = Variable(str, title="Prompt")
    duration = Variable(int, title="Duration")
    model = MusicgenModel()
    model.load()
    output = model.predict(prompt, duration)
    p.output(output)

pipeline = p.get_pipeline()
