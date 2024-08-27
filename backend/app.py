from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from logging.handlers import RotatingFileHandler
from time import strftime
import traceback
from werkzeug.utils import secure_filename
import os 
import torch
import transformers
import torchaudio
import io

app = Flask(__name__)
CORS(app)

# Set the upload folder
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)



'''@app.route('/api/data', methods=['POST'])
def process_data():
    data = request.json
    # Do something with the data
    return jsonify({'status': 'success', 'data_received': data})'''


processor = Wav2Vec2Processor.from_pretrained("KBLab/wav2vec2-large-voxrex-swedish")
model = Wav2Vec2ForCTC.from_pretrained("KBLab/wav2vec2-large-voxrex-swedish")
#model.eval()
resampler = torchaudio.transforms.Resample(48_000, 16_000)

@app.route('/api/upload', methods=['POST'])
def transcribe_audio():
    try:
        # Ensure an audio file is provided
        if 'audio' not in request.files:
            raise ValueError("No audio file found in the request.")

        # Get the file from the request
        audio_file = request.files['audio']

        # Read the audio file bytes
        audio_bytes = io.BytesIO(audio_file.read())

        # Load the audio file with torchaudio
        waveform, sample_rate = torchaudio.load(audio_bytes, format="wav")

        # Resample the audio to 16kHz if necessary
        if sample_rate != 16000:
            waveform = resampler(waveform)

        # Process the audio input
        input_data = waveform.squeeze().numpy()
        inputs = processor(input_data, sampling_rate=16_000, return_tensors="pt", padding=True)

        # Perform the model inference
        with torch.no_grad():
            logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

        # Decode the prediction
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)

        # Log the successful request
        logging.info(f"Transcription successful for file: {audio_file.filename}")

        # Return the result as JSON
        return jsonify({'transcription': transcription[0]}), 200

    except Exception as e:
        # Log the error
        logging.error(f"Error processing file: {str(e)}")
        return jsonify({'error': str(e)}), 400

'''async def predict(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    audio_b = io.BytesIO(audio_bytes)
    
    audio_input, _ = librosa.load(audio_b , sr=16000)
    inputs = processor(audio_input, return_tensors="pt", padding="longest")
    
    waveform, sample_rate = torchaudio.load(audio_b)
    input_data = resampler(waveform).squeeze().numpy()[:2]

    inputs = processor(input_data, sampling_rate=16_000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(input_values=inputs.input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return {"transcription": transcription}'''

'''def process_data():
    if 'audio' not in request.files:
        return jsonify({'status': 'error', 'message': 'No audio file part in the request'}), 400

    file = request.files['audio']
    
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400

    if file and file.filename.endswith(('mp3', 'wav', 'ogg')):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify({'status': 'success', 'filename': filename}), 200

    return jsonify({'status': 'error', 'message': 'Invalid file type'}), 400
'''

@app.after_request
def after_request(response):
    timestamp = strftime('[%Y-%b-%d %H:%M]')
    logger.error('%s %s %s %s %s %s', timestamp, request.remote_addr, request.method, request.scheme, request.full_path, response.status)
    return response

@app.errorhandler(Exception)
def exceptions(e):
    tb = traceback.format_exc()
    timestamp = strftime('[%Y-%b-%d %H:%M]')
    logger.error('%s %s %s %s %s 5xx INTERNAL SERVER ERROR\n%s', timestamp, request.remote_addr, request.method, request.scheme, request.full_path, tb)
    return e.status_code

if __name__ == '__main__':
    handler = RotatingFileHandler('/app/logs/app.log', maxBytes=100000, backupCount=3)
    logger = logging.getLogger('tdm')
    logger.setLevel(logging.ERROR)
    logger.addHandler(handler)
    
    app.run(host='0.0.0.0', port=8000)



