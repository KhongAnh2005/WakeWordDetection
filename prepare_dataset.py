import os
import numpy as np
import librosa


# CẤU HÌNH DATASET
DATASET_PATH = "dataset"

SR = 16000          # sample rate chuẩn cho speech
DURATION = 2        # chuẩn hóa mọi audio thành 2 giây
TARGET_LEN = SR * DURATION

# các định dạng audio được hỗ trợ
AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac")


# LOAD AUDIO + CHUẨN HÓA
def load_and_normalize(file_path):

    # load audio và resample về 16kHz
    audio, sr = librosa.load(file_path, sr=SR, mono=True)

    # normalize biên độ về [-1,1]
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    # chuẩn hóa độ dài audio về 2 giây
    if len(audio) < TARGET_LEN:
        audio = np.pad(audio, (0, TARGET_LEN - len(audio)))
    else:
        audio = audio[:TARGET_LEN]

    return audio


# TRÍCH XUẤT FEATURE (MFCC)
def extract_features(audio):

    # tính MFCC matrix
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SR,
        n_mfcc=13,
        n_fft=512,
        hop_length=160,
        window="hann"
    )

    # thống kê MFCC theo thời gian
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # delta MFCC (tốc độ thay đổi)
    delta = librosa.feature.delta(mfcc)
    delta_mean = np.mean(delta, axis=1)

    # tạo vector feature (13 + 13 + 13 = 39)
    features = np.concatenate([mfcc_mean, mfcc_std, delta_mean])

    return features


# TẠO DATASET
X = []   # feature matrix
y = []   # labels

labels = ["non_wake", "wake"]


# ĐỌC TẤT CẢ AUDIO TRONG DATASET
for label_index, label_name in enumerate(labels):

    folder_path = os.path.join(DATASET_PATH, label_name)

    for file in os.listdir(folder_path):

        # bỏ qua file không phải audio
        if not file.lower().endswith(AUDIO_EXTENSIONS):
            continue

        file_path = os.path.join(folder_path, file)

        try:

            audio = load_and_normalize(file_path)
            features = extract_features(audio)

            X.append(features)
            y.append(label_index)

        except Exception as e:
            print("Error:", file_path)


# CHUYỂN SANG NUMPY ARRAY
X = np.array(X)
y = np.array(y)


# SHUFFLE DATASET
indices = np.arange(len(X))
np.random.shuffle(indices)

X = X[indices]
y = y[indices]


# LƯU DATASET
np.save("X.npy", X)
np.save("y.npy", y)

print("Dataset saved")