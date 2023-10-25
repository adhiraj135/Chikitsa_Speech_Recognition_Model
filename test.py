import joblib
import whisper

model = joblib.load("Speech_Recognition_Model.h5")

results=model.transcribe("Wednesday at 12-02 PM.m4a",fp16=False)

print(results["text"])