import os
import numpy as np
import librosa
import soundfile as sf


# CẤU HÌNH
SR = 16000
DATASET_PATH = "dataset"
AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac")

# số bản augment sinh ra từ mỗi file gốc
N_AUGMENTS = 10


# THÊM NHIỄU TRẮNG
def add_noise(audio, factor=0.005):
    noise = np.random.randn(len(audio))
    return audio + factor * noise


# THAY ĐỔI TỐC ĐỘ NÓI (không đổi pitch)
def time_stretch(audio, rate=None):
    if rate is None:
        rate = np.random.uniform(0.85, 1.15)
    return librosa.effects.time_stretch(audio, rate=rate)


# THAY ĐỔI CAO ĐỘ
def pitch_shift(audio, steps=None):
    if steps is None:
        steps = np.random.uniform(-2, 2)
    return librosa.effects.pitch_shift(audio, sr=SR, n_steps=steps)


# THAY ĐỔI ÂM LƯỢNG
def change_volume(audio, factor=None):
    if factor is None:
        factor = np.random.uniform(0.7, 1.3)
    return audio * factor


# AUGMENT 1 FILE
def augment_file(file_path, output_folder, n=N_AUGMENTS):

    audio, _ = librosa.load(file_path, sr=SR, mono=True)

    base_name = os.path.splitext(os.path.basename(file_path))[0]

    for i in range(n):

        aug = audio.copy()

        # áp dụng ngẫu nhiên 1-3 kỹ thuật augment
        ops = np.random.choice(["noise", "stretch", "pitch", "volume"], 
                                size=np.random.randint(1, 4), replace=False)

        if "noise" in ops:
            aug = add_noise(aug)

        if "stretch" in ops:
            aug = time_stretch(aug)

        if "pitch" in ops:
            aug = pitch_shift(aug)

        if "volume" in ops:
            aug = change_volume(aug)

        # normalize lại sau augment
        max_val = np.max(np.abs(aug))
        if max_val > 0:
            aug = aug / max_val

        # lưu file augmented
        out_path = os.path.join(output_folder, f"{base_name}_aug{i}.wav")
        sf.write(out_path, aug, SR)

    print(f"Augmented {n} files from: {os.path.basename(file_path)}")


# CHẠY AUGMENT CHO CẢ 2 CLASS
for label in ["wake", "non_wake"]:

    folder = os.path.join(DATASET_PATH, label)

    files = [f for f in os.listdir(folder) 
             if f.lower().endswith(AUDIO_EXTENSIONS)
             and "_aug" not in f]  # bỏ qua file augment cũ

    print(f"\n--- Augmenting class: {label} ({len(files)} original files) ---")

    for file in files:
        augment_file(os.path.join(folder, file), folder)

print("\nDone! Now re-run prepare_dataset.py and train_model.py")