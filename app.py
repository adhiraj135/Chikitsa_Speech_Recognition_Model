from flask import Flask, request, render_template,jsonify
import base64
import whisper
import time
import joblib
from Bio_Epidemiology_NER.bio_recognizer import ner_prediction
import warnings
import os
warnings.filterwarnings('ignore')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload():
    audio_file = request.files['audio_file']
    if audio_file:
        audio_path = os.path.join('static/uploaded audios', audio_file.filename)
        audio_file.save(audio_path)


    model=joblib.load("Speech_Recognition_Model.h5")
    time.sleep(3)
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = {"fp16": False, "language": "hi", "task": "translate"}
    result = whisper.decode(model, mel, **options)
    text = result.text
    detected_text = text + ""
    print(detected_text)
    ner = ner_prediction(corpus=detected_text, compute='cpu')
    medicines = []
    for i in ner["value"][ner["entity_group"] == "Medication"].ravel():
        medicines.append(i)
    return render_template('home.html', detected_text=medicines)


if __name__=='__main__':
    app.run(host='0.0.0.0',port=8080)
