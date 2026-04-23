from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
import os
import uuid

app = FastAPI(title="Pocket TTS API")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/tts")
async def tts(text: str, voice: str = "default"):
    try:
        import torch
        from pocket_tts import AutoModelForTextToSpeech, AutoProcessor
        
        model_name = "kyutai/pocket-tts"
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForTextToSpeech.from_pretrained(model_name, trust_remote_code=True)
        
        inputs = processor(text=text, return_tensors="pt")
        with torch.no_grad():
            audio = model.generate(**inputs)
        
        import numpy as np
        import wave
        
        audio_np = audio.cpu().numpy().squeeze()
        if len(audio_np.shape) == 0:
            audio_np = np.array([audio_np.item()])
        sample_rate = 24000
        
        tmp_path = f"/tmp/tts_{uuid.uuid4().hex}.wav"
        with wave.open(tmp_path, "w") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sample_rate)
            audio_int = (np.clip(audio_np, -1, 1) * 32767).astype(np.int16)
            f.writeframes(audio_int.tobytes())
        
        with open(tmp_path, "rb") as f:
            audio_data = f.read()
        os.unlink(tmp_path)
        
        return Response(
            content=audio_data,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=tts.wav"}
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)

