from fastapi import FastAPI, File, UploadFile, Request
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import librosa
from fastapi.responses import JSONResponse

from fastapi.middleware.cors import CORSMiddleware
#from starlette.middleware import Middleware
#from starlette.middleware.cors import CORSMiddleware
import io
import torchaudio
from datasets import load_dataset
import IPython
import logging 


#from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(debug=True)

#origins = ["*"]


'''app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)'''

origins = [
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows the specified origin(s)
    allow_credentials=True,  # Allows cookies and credentials to be sent
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Log request details
    logging.info(f"Request: {request.method} {request.url}")
    
    # Process request
    response = await call_next(request)
    
    # Log response details
    logging.info(f"Response status code: {response.status_code}")
    
    return response


# Configure logging
logging.basicConfig(
    filename='app.log',  # Log file location
    level=logging.INFO,  # Log level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)

# Load the processor and model
processor = Wav2Vec2Processor.from_pretrained("KBLab/wav2vec2-large-voxrex-swedish")
model = Wav2Vec2ForCTC.from_pretrained("KBLab/wav2vec2-large-voxrex-swedish")
#model.eval()
resampler = torchaudio.transforms.Resample(48_000, 16_000)


def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = resampler(speech_array).squeeze().numpy()
    return batch


# Endpoint for transcription
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the file
        audio_bytes = await file.read()
        audio_b = io.BytesIO(audio_bytes)
        
        # Load and process audio
        waveform, sample_rate = torchaudio.load(audio_b)
        input_data = resampler(waveform).squeeze().numpy()#[:2]
        inputs = processor(input_data, sampling_rate=16_000, return_tensors="pt", padding=True)
        
        # Make prediction
        with torch.no_grad():
            logits = model(input_values=inputs.input_values).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        
        # Log transcription result
        logging.info(f"Transcription: {transcription}")
        
        return {"transcription": transcription}
    
    except Exception as e:
        # Log any exception that occurs
        logging.error(f"Error during prediction: {str(e)}")
        print(str(e))
        return JSONResponse(content={"error": "An error occurred during processing."}, status_code=500)
