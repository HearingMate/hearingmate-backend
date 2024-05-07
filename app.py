# -*- coding: utf-8 -*-


from flask import Flask, request, jsonify
import librosa
import numpy as np
from skimage.transform import resize
from tensorflow import keras
import base64
import io
from tensorflow import keras
import os
import tensorflow as tf



app = Flask(__name__)

model_save_path = "model"
audio_file_path = "sound/audio.wav"


def loadModel():
    global model
    model = tf.saved_model.load(model_save_path)

loadModel()



def decode_and_save_audio(base64_audio, audio_file_path):
    audio_data = base64.b64decode(base64_audio)

    if os.path.exists(audio_file_path):
        os.remove(audio_file_path)

    with open(audio_file_path, "wb") as f:
        f.write(audio_data)

    return audio_file_path


def resize_spectrogram(spec, target_shape):
    return resize(spec, target_shape, anti_aliasing=True)

def preprocess_audio(file_path, target_shape):
    # Load the audio file
    audio, sr = librosa.load(file_path)

    # Convert the audio file to the appropriate format (here, mel spectrogram is used as an example)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512, n_mels=90)
    log_mel = librosa.power_to_db(mel_spectrogram)

    # Convert the spectrogram to 3D
    spectrogram_3d = log_mel[..., np.newaxis]

    # Resize the spectrogram to the target shape
    resized_spectrogram = resize(spectrogram_3d, target_shape, anti_aliasing=True)

    return resized_spectrogram


def make_prediction(preprocessed_input):
    # Adjust the shape as expected by the model (e.g., (None, 300, 200, 1))
    input_for_model = np.expand_dims(preprocessed_input, axis=0)

    # Make predictions
    predictions = model(input_for_model)

    return predictions




@app.route("/predict", methods=["POST"] )
def defineCategory():
    
    
    data = request.json
    base64_audio = data.get('audio')
    
    if base64_audio :
            print("Audio Received!")
    else:
            return "Audio Not Received!"
        
    file_path = decode_and_save_audio(base64_audio,audio_file_path)
        
    target_shape = (300, 200)  # Target shape
        
    preprocessed_input = preprocess_audio(file_path, target_shape)

    model_predictions = make_prediction(preprocessed_input)

    model_predictions = [prediction.numpy().tolist() for prediction in model_predictions]

    print("Predictions:", model_predictions)

    return jsonify({"predictions": model_predictions})



if __name__ == "__main__":
    print("Listening port")
    app.run(debug=True, port=8000)