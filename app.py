from flask import Flask, request
from tensorflow import keras
import librosa
import numpy as np
from skimage.transform import resize


app = Flask(__name__)

model_save_path = "/model"

def loadModel():
    global model
    model = keras.models.load_model(model_save_path)

loadModel() 


def resize_spectrogram(spec, target_shape):
    return resize(spec, target_shape, anti_aliasing=True)

def preprocess_audio(audio_file_path, target_shape):
    # Load the audio file
    audio, sr = librosa.load(audio_file_path)

    # Convert the audio file to the appropriate format (here, mel spectrogram is used as an example)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512, n_mels=90)
    log_mel = librosa.power_to_db(mel_spectrogram)

    # Convert the spectrogram to 3D
    spectrogram_3d = log_mel[..., np.newaxis]

    # Resize the spectrogram to the target shape
    resized_spectrogram = resize(spectrogram_3d, target_shape, anti_aliasing=True)

    return resized_spectrogram



def make_prediction(model, preprocessed_input):
    # Adjust the shape as expected by the model (e.g., (None, 300, 200, 1))
    input_for_model = np.expand_dims(preprocessed_input, axis=0)

    # Make predictions
    predictions = model.predict(input_for_model)

    return predictions




@app.route("/predict", methods=["POST"] )
def defineCategory():
    
    target_shape = (300, 200)  # Target shape
    data = request.get_json()
    
    if data :
            print("Audio Received!")
    else:
            return "Audio Not Received!"
        
    preprocessed_input = preprocess_audio(data, target_shape)

    model_predictions = make_prediction(model, preprocessed_input)

    print("Predictions:", model_predictions)



if __name__ == "__main__":
    print("Listening port")
    app.run(debug=True, port=8000)