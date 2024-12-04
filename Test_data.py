import os
import cv2  # type: ignore
import librosa  # type: ignore
import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
import pickle
import sys

sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

model = tf.keras.models.load_model('voice_pathology_model.h5')

with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

pathology_labels = ['Pathology']
healthy_labels = ['Healthy']

def preprocess_single_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        duration = len(y) / sr

        D = librosa.stft(y)
        spectrogram = librosa.amplitude_to_db(np.abs(D))

        spectrogram_resized = cv2.resize(spectrogram, (40, 1025))
        spectrogram_resized = np.expand_dims(spectrogram_resized, axis=-1)
        spectrogram_resized = np.expand_dims(spectrogram_resized, axis=0)

        return spectrogram_resized, sr, duration
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None, None, None

test_audio_paths = [
    r'test_wav\Healthy_voice.wav', 
]
gender_input = "Male" 

predicted_labels = []
male_correct_predictions = 0
female_correct_predictions = 0
male_total = 0
female_total = 0

for file_path in test_audio_paths:
    if os.path.exists(file_path):
        input_audio, sr, duration = preprocess_single_audio(file_path)
        if input_audio is not None:
            prediction = model.predict(input_audio)

            predicted_class_index = np.argmax(prediction, axis=1)[0]
            predicted_label = str(label_encoder.inverse_transform([predicted_class_index])[0])

            predicted_labels.append(predicted_label)

            if gender_input == 'Male':
                male_total += 1
                if predicted_label in pathology_labels:
                    male_correct_predictions += 1
            elif gender_input == 'Female':
                female_total += 1
                if predicted_label in healthy_labels:
                    female_correct_predictions += 1

            print(f"\n{'Prediction Details':^50}")
            print("=" * 50)
            print(f"{'File Name':<40} {file_path}")
            print(f"{'Predicted Label':<40} {predicted_label}")
            print(f"{'Accuracy':<40} {prediction[0][predicted_class_index]:<15.2f}")
            print("=" * 50)

            print(f"\n{'Audio Summary':^50}")
            print("=" * 50)
            print(f"{'Path':<30} {file_path}")
            print(f"{'Sample Rate':<30} {sr}")
            print(f"{'Duration':<30} {duration:.2f}s")
            print("=" * 50)
        else:
            print(f"Failed to preprocess the file: {file_path}")
            predicted_labels.append(None)
    else:
        print(f"File does not exist: {file_path}")
        predicted_labels.append("File Missing")

male_accuracy = (male_correct_predictions / male_total * 100) if male_total > 0 else 0
female_accuracy = (female_correct_predictions / female_total * 100) if female_total > 0 else 0

print("\n" + "=" * 50)
print(f"{'### **Gender-Based Accuracy** ###':^50}")
print("=" * 50)

if gender_input == "Male":
    print(f"{'Male Accuracy':<40} {prediction[0][predicted_class_index]:<15.2f}")
elif gender_input == "Female":
    print(f"{'Female Accuracy':<40} {prediction[0][predicted_class_index]:<15.2f}")
else:
    print("Invalid gender input. Please use 'Male' or 'Female'.")
print("=" * 50)

print("\n" + "=" * 50)
print(f"{'### **Debugging Information** ###':^50}")
print("=" * 50)
print(f"{'Predicted Labels:':<30} {predicted_labels}")
print("=" * 50)
