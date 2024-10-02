import uvicorn
from dotenv import load_dotenv
from configs.config import Config
from fastapi import FastAPI, HTTPException, Path, Request, Form
from fastapi.responses import Response
from infer.modules.vc.modules import VC
import os
import io
import tempfile
import numpy as np
import soundfile as sf
import boto3

load_dotenv()
config = Config()
app = FastAPI()

def create_wav_response(wav_opt):
    wav_io = io.BytesIO()
    wav_io.write(b'RIFF')
    wav_io.write((36 + len(wav_opt[1])*2).to_bytes(4, 'little'))
    wav_io.write(b'WAVE')
    wav_io.write(b'fmt ')
    wav_io.write((16).to_bytes(4, 'little'))
    wav_io.write((1).to_bytes(2, 'little'))
    wav_io.write((1).to_bytes(2, 'little'))
    wav_io.write(wav_opt[0].to_bytes(4, 'little'))
    wav_io.write((wav_opt[0] * 2).to_bytes(4, 'little'))
    wav_io.write((2).to_bytes(2, 'little'))
    wav_io.write((16).to_bytes(2, 'little'))
    wav_io.write(b'data')
    wav_io.write((len(wav_opt[1])*2).to_bytes(4, 'little'))
    wav_io.write(wav_opt[1].astype(np.int16).tobytes())
    return Response(content=wav_io.getvalue(), media_type="audio/wav")

def perform_vc(vc, input_path, model_filename, index):
    vc.get_vc(model_filename)
    _, wav_opt = vc.vc_single(
        0, input_path, 0, None, "harvest", f"./opt/{index}", None,
        0.66, 3, 0, 1, 0.33
    )
    return wav_opt

@app.post("/text2voice2voice/{model_filename}/{index}")
async def text2voice2voice(
    request: Request,
    model_filename: str = Path(..., description="Filename of the .pth model"),
    index: str = Path(..., description="Filename of the index file"),
    text: str = Form(..., description="Text to convert to speech"),
    voice_id: str = Form(default="Mizuki", description="Amazon Polly voice ID")
):
    try:
        polly_client = boto3.Session(region_name=os.getenv('AWS_REGION', 'ap-northeast-1')).client('polly')
        response = polly_client.synthesize_speech(Text=text, OutputFormat='pcm', VoiceId=voice_id, SampleRate='16000')

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav_file:
            tmp_wav_path = tmp_wav_file.name
            sf.write(tmp_wav_path, np.frombuffer(response['AudioStream'].read(), dtype=np.int16), 16000, format='WAV')

        vc = VC(config)
        wav_opt = perform_vc(vc, tmp_wav_path, model_filename, index)

        os.remove(tmp_wav_path)
        return create_wav_response(wav_opt)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice2voice/{model_filename}/{index}")
async def voice2voice(
    request: Request,
    model_filename: str = Path(..., description="Filename of the .pth model"),
    index: str = Path(..., description="Filename of the index file")
):
    try:
        wav_data = await request.body()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(wav_data)
            tmp_file_path = tmp_file.name

        vc = VC(config)
        wav_opt = perform_vc(vc, tmp_file_path, model_filename, index)

        os.remove(tmp_file_path)
        return create_wav_response(wav_opt)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, port=8080)
