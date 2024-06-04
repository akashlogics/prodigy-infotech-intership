import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, LeakyReLU
from sklearn.metrics import classification_report
import seaborn as sns

# Load and preprocess images
folders_names = [f'C:\\Users\\user\\Desktop\\archive (3)\\leapGestRecog\\0{i}' for i in range(10)]
files_names = ['01_palm', '02_l', '03_fist', '04_fist_moved', '05_thumb']

X = []
y = []

for folder in folders_names:
    for file in files_names:
        path = os.path.join(folder, file)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            X.append(img_array)
            y.append(int(folder[-1]))

X = np.array(X) / 255
y = np.array(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the model
model = Sequential([
    Flatten(input_shape=(240, 640)),
    Dense(64),
    LeakyReLU(alpha=0.1),
    Dense(32),
    LeakyReLU(alpha=0.1),
    Dense(16),
    LeakyReLU(alpha=0.1),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=3, validation_split=0.1, batch_size=32, verbose=2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)

# Make predictions and generate classification report
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

conf_mat = tf.math.confusion_matrix(labels=y_test, predictions=y_pred).numpy()

print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Visualize confusion matrix
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='bone')