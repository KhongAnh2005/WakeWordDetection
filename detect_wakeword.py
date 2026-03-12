import os
import numpy as np
import librosa
import joblib
import sounddevice as sd


# CẤU HÌNH
SR = 16000
DURATION = 2
TARGET_LEN = SR * DURATION

# ngưỡng xác suất để xác nhận wake word, tránh false positive
CONFIDENCE_THRESHOLD = 0.5

# tỉ lệ overlap giữa các cửa sổ khi nghe realtime (0.5 = 50%)
OVERLAP = 0.5
HOP_LEN = int(TARGET_LEN * (1 - OVERLAP))

AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac")


# KIỂM TRA MODEL TỒN TẠI TRƯỚC KHI LOAD
if not os.path.exists("wakeword_model.pkl") or not os.path.exists("scaler.pkl"):
    print("ERROR: Model not found. Run train_model.py first.")
    exit()


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

    # dùng predict_proba thay predict để lọc theo ngưỡng tin cậy
    proba = model.predict_proba(features)[0][1]

    return proba


# MODE 1: AUDIO FILE
def detect_from_file(file_path):

    audio, _ = librosa.load(file_path, sr=SR, mono=True)

    # nếu file ngắn hơn hoặc bằng 1 window thì detect thẳng, không cần sliding
    if len(audio) <= TARGET_LEN:
        proba = detect(audio)
        if proba >= CONFIDENCE_THRESHOLD:
            print(f"Wake word detected! (confidence: {proba:.2f})")
        else:
            print(f"No wake word detected. (confidence: {proba:.2f})")
        return

    # dùng sliding window với overlap để không bỏ sót wake word ở ranh giới đoạn
    detected = False

    start = 0
    while start + TARGET_LEN <= len(audio):

        window = audio[start: start + TARGET_LEN]

        proba = detect(window)

        if proba >= CONFIDENCE_THRESHOLD:
            time_sec = start / SR
            print(f"Wake word detected at {time_sec:.2f}s (confidence: {proba:.2f})")
            detected = True

        start += HOP_LEN

    if not detected:
        print("No wake word detected.")


# MODE 2: MICROPHONE REALTIME
def detect_from_mic():

    count = 0

    # buffer giữ lại nửa cửa sổ trước để tạo sliding window liên tục
    buffer = np.zeros(TARGET_LEN, dtype="float32")

    print("Listening... Press Ctrl+C to stop")

    while True:

        # chỉ record phần mới (HOP_LEN samples), ghép với buffer cũ
        new_audio = sd.rec(
            HOP_LEN,
            samplerate=SR,
            channels=1,
            dtype="float32"
        )

        sd.wait()

        new_audio = new_audio.flatten()

        # cập nhật buffer: dịch trái và ghép audio mới vào cuối
        buffer = np.concatenate([buffer[HOP_LEN:], new_audio])

        proba = detect(buffer.copy())

        if proba >= CONFIDENCE_THRESHOLD:
            count += 1
            print(f"Wake word detected! Total: {count} (confidence: {proba:.2f})")
        else:
            print(f"No wake word (confidence: {proba:.2f})")


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