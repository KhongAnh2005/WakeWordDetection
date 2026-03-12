import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# LOAD DATASET
X = np.load("X.npy")
y = np.load("y.npy")

print("Dataset shape:", X.shape)


# CHIA TRAIN / TEST
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# SCALE FEATURE
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# TẠO MODEL SVM
model = SVC(
    kernel="rbf",
    probability=True
)


# TRAIN MODEL
model.fit(X_train, y_train)


# TEST MODEL
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)


# LƯU MODEL
joblib.dump(model, "wakeword_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model saved")