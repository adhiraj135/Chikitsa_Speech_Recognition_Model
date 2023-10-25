from flask import Flask, render_template, request, jsonify
import os
import base64
import joblib
import whisper
import time
from flask_cors import CORS
from Bio_Epidemiology_NER.bio_recognizer import ner_prediction
import warnings

warnings.filterwarnings('ignore')



application=Flask(__name__)
CORS(application)


@application.route('/')
def index():
    return render_template('index.html')

@application.route('/save_audio', methods=['POST'])
def save_audio():
    try:
        audio_file = request.json['audio_data']

        # Decode the base64 audio data and save it to a file
        if audio_file:
            audio_path = os.path.join('static/uploaded audios', audio_file.filename)
            audio_file.save(audio_path)
        model = joblib.load("Speech_Recognition_Model.h5")
        # result=model.transcribe('static/recorded audios/recorded_audio.wav')
        # detected_text=result["text"]
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

        # ner=ner_prediction(corpus=detected_text, compute='cpu')
        # medicines = []
        # for i in ner["value"][ner["entity_group"] == "Medication"].ravel():
        #    medicines.append(i)
        # print(medicines)

        return jsonify({'message': 'Audio saved successfully','text':detected_text})

    except Exception as e:
        raise e


@application.route('/get_transcription', methods=['GET'])
def get_transcription():
    try:
        with open('Wednesday at 12-02 PM.m4a', 'rb') as audio_file:
            audio_data = base64.b64encode(audio_file.read()).decode()
        data = {'audio_data': audio_data}
        model = joblib.load("Speech_Recognition_Model.h5")
        # result=model.transcribe('static/recorded audios/recorded_audio.wav')
        # detected_text=result["text"]
        time.sleep(3)
        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio('Wednesday at 12-02 PM.m4a')
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
        ner = ner_prediction(corpus=detected_text, compute='cpu')
        medicines = []
        for i in ner["value"][ner["entity_group"] == "Medication"].ravel():
            medicines.append(i)
        return jsonify({'message': detected_text, 'text': medicines})
    except Exception as e:
        return jsonify({'error': 'Error transcribing audio', 'message': str(e)})

if __name__ == '__main__':
    application.run(host='0.0.0.0',port=6060)



