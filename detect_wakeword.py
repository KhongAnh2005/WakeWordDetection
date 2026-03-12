import numpy as np
import librosa
import joblib
import sounddevice as sd


# CẤU HÌNH
SR = 16000
DURATION = 2
TARGET_LEN = SR * DURATION

AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac")


# LOAD MODEL
model = joblib.load("wakeword_model.pkl")
scaler = joblib.load("scaler.pkl")


# CHUẨN HÓA AUDIO
def normalize_audio(audio):

    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    if len(audio) < TARGET_LEN:
        audio = np.pad(audio, (0, TARGET_LEN - len(audio)))
    else:
        audio = audio[:TARGET_LEN]

    return audio


# TRÍCH XUẤT FEATURE
def extract_features(audio):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SR,
        n_mfcc=13,
        n_fft=512,
        hop_length=160,
        window="hann"
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    delta = librosa.feature.delta(mfcc)
    delta_mean = np.mean(delta, axis=1)

    features = np.concatenate([mfcc_mean, mfcc_std, delta_mean])

    return features.reshape(1, -1)


# DETECT WAKE WORD
def detect(audio):

    audio = normalize_audio(audio)

    features = extract_features(audio)

    features = scaler.transform(features)

    prediction = model.predict(features)

    return prediction[0]


# MODE 1: AUDIO FILE
def detect_from_file(file_path):

    audio, _ = librosa.load(file_path, sr=SR, mono=True)

    result = detect(audio)

    if result == 1:
        print("Wake word detected!")
    else:
        print("No wake word detected.")


# MODE 2: MICROPHONE REALTIME
def detect_from_mic():

    count = 0

    print("Listening... Press Ctrl+C to stop")

    while True:

        audio = sd.rec(
            TARGET_LEN,
            samplerate=SR,
            channels=1,
            dtype="float32"
        )

        sd.wait()

        audio = audio.flatten()

        result = detect(audio)

        if result == 1:
            count += 1
            print("Wake word detected! Total:", count)
        else:
            print("No wake word")


# MAIN MENU
print("Choose input mode:")
print("1 - Audio file")
print("2 - Microphone")

mode = input("Enter choice: ")

if mode == "1":

    file_path = input("Enter audio file path: ")

    detect_from_file(file_path)

elif mode == "2":

    detect_from_mic()

else:

    print("Invalid option")