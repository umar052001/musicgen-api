import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import random

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
 

class Prompt(BaseModel):
    prompt: str

@app.post("/generate-audio")
async def generate_audio(prompt: Prompt):
    model = MusicGen.get_pretrained('large')
    model.set_generation_params(duration=8)  # generate 8 seconds.

    descriptions = [prompt.prompt]
    wav = model.generate(descriptions)

    return wav
