import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Load data
data = pd.read_csv('cleaned_output.csv')

# Load images and preprocess
images = []
for img_path in data['image_path']:
    img = cv2.imread(img_path)
    img = cv2.resize(img, (100, 100))  # Resize images to a fixed size
    images.append(img)
images = np.array(images)

# Normalize features
scaler = MinMaxScaler()
images = images / 255.0  # Normalize pixel values to range [0, 1]

# Define target variables
y = data[['contrast', 'entropy', 'localContrast', 'localEntropy', 'saturation']].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(images, y, test_size=0.2, random_state=42)

# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(5)  # Output layer with 5 neurons for 5 target variables
])

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
