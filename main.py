from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
import time
import torch
from chatterbox.tts import ChatterboxTTS


app = FastAPI()
# Config
VOICE_DIR = "voices"
os.makedirs(VOICE_DIR, exist_ok=True)

# CORS (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

voices = {
    "emily": 'voices/emily.wav',
    "danial":'voices/denial.wav'
}
# Build voices dict
def load_voices():
    return {
        os.path.splitext(f)[0]: os.path.join(VOICE_DIR, f)
        for f in os.listdir(VOICE_DIR)
        if f.endswith(".wav")
    }
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model = ChatterboxTTS.from_pretrained(device=device)

class TTSRequest(BaseModel):
    input: str
    voice: str = "emily"

@app.post("/v1/audio/stream_pcm")
async def stream_pcm_audio(req: TTSRequest):
    print(f"\n📥 Text received: {req.input}")

    def pcm_stream_generator():
        total_samples = 0
        chunk_count = 0
        chunk_times = []
        start_time = time.time()
        first_chunk_time = None

        try:
            for audio_chunk, _ in model.generate_stream(
                text=req.input,
                chunk_size=50,
                exaggeration=0.3,
                temperature=0.2,
                cfg_weight=0.5,
                seed = 123,
                print_metrics=False
            ):
                now = time.time()

                if first_chunk_time is None:
                    first_chunk_time = now
                    print(f"🚀 TTFT (server): {first_chunk_time - start_time:.3f}s")

                chunk_count += 1
                duration = audio_chunk.shape[-1] / model.sr
                total_samples += audio_chunk.shape[-1]
                chunk_times.append(now)
                print(f"📦 Chunk {chunk_count}: {duration:.3f}s at {now - start_time:.2f}s")

                yield audio_chunk.squeeze().numpy().astype("float32").tobytes()

        finally:
            end_time = time.time()
            total_time = end_time - start_time
            audio_duration = total_samples / model.sr
            rtf = total_time / audio_duration if audio_duration > 0 else float('inf')

            print("\n🧾 Server-side Metrics:")
            print(f"🕒 Total Time: {total_time:.2f}s")
            print(f"🔉 Audio Duration: {audio_duration:.2f}s")
            print(f"⚡ RTF (server): {rtf:.3f}")
            print(f"📦 Total Chunks: {chunk_count}")
            print("=" * 50)

    return StreamingResponse(pcm_stream_generator(), media_type="audio/L16")

@app.get("/v1/voices")
async def get_voices():
    return list(load_voices().keys())

@app.post("/v1/voices/register")
async def register_voice(file: UploadFile, name: str = Form(...)):
    ext = os.path.splitext(file.filename)[1]
    if ext.lower() != ".wav":
        return JSONResponse({"error": "Only .wav files are supported"}, status_code=400)
    path = os.path.join(VOICE_DIR, f"{name}.wav")
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"message": f"Voice '{name}' registered successfully."}

@app.post("/v1/audio/stream_vc")
async def stream_pcm_clone(req: TTSRequest):
    voices = load_voices()
    if req.voice not in voices:
        return JSONResponse(content={"error": f"Voice '{req.voice}' not found."}, status_code=404)

    print(f"\n📥 Text received: {req.input}")
    print(f"🎙️ Voice selected: {req.voice}")

    def pcm_stream_generator():
        total_samples = 0
        chunk_count = 0
        chunk_times = []
        start_time = time.time()
        first_chunk_time = None

        try:
            for audio_chunk, _ in model.generate_stream(
                text=req.input,
                chunk_size=50,
                audio_prompt_path=voices[req.voice],
                exaggeration=0.5,
                temperature=0.2,
                cfg_weight=0.5,
                print_metrics=True
            ):
                now = time.time()
                if first_chunk_time is None:
                    first_chunk_time = now
                    print(f"🚀 TTFT (server): {first_chunk_time - start_time:.3f}s")

                chunk_count += 1
                duration = audio_chunk.shape[-1] / model.sr
                total_samples += audio_chunk.shape[-1]
                chunk_times.append(now)
                print(f"📦 Chunk {chunk_count}: {duration:.3f}s at {now - start_time:.2f}s")
                yield audio_chunk.squeeze().numpy().astype("float32").tobytes()
        except Exception as e:
            print(f"❗ Error during audio generation: {e}")
            return JSONResponse(content={"error": str(e)}, status_code=500)
        finally:
            end_time = time.time()
            total_time = end_time - start_time
            audio_duration = total_samples / model.sr
            rtf = total_time / audio_duration if audio_duration > 0 else float('inf')

            print("\n🧾 Server-side Metrics:")
            print(f"🕒 Total Time: {total_time:.2f}s")
            print(f"🔉 Audio Duration: {audio_duration:.2f}s")
            print(f"⚡ RTF (server): {rtf:.3f}")
            print(f"📦 Total Chunks: {chunk_count}")
            print("=" * 50)

    return StreamingResponse(pcm_stream_generator(), media_type="audio/L16")
