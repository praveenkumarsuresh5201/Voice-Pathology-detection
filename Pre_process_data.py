import os
import cv2 # type: ignore # type: ignore
import librosa # type: ignore
import numpy as np # type: ignore
import tensorflow as tf # type: ignore
import pickle
import sys

sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

model = tf.keras.models.load_model('voice_pathology_model.h5')

with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

def preprocess_single_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        D = librosa.stft(y)
        spectrogram = librosa.amplitude_to_db(np.abs(D))
        spectrogram_resized = cv2.resize(spectrogram, (40, 1025))
        spectrogram_resized = np.expand_dims(spectrogram_resized, axis=-1)
        spectrogram_resized = np.expand_dims(spectrogram_resized, axis=0)  
        return spectrogram_resized
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

input_audio_path = 'test_wav\Mohan MP Kootturavu Nagar.wav'

input_audio = preprocess_single_audio(input_audio_path)

if input_audio is not None:
    prediction = model.predict(input_audio)
    
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_label = label_encoder.inverse_transform([predicted_class_index])
    
    try:
        print(f"Predicted label: {predicted_label[0]}")
    except UnicodeEncodeError:
        print("Error encoding the output label. The label might contain unsupported characters.")
else:
    print("Failed to preprocess the audio file.")