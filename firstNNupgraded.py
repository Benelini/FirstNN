from sklearn.datasets import load_breast_cancer
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import load_model

data = load_breast_cancer()
X = data.data
y = data.target

# Data Preprocessing: Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Improved Model Architecture
model = tf.keras.Sequential([
    layers.Dense(30, activation="relu", input_shape=(X_train.shape[1],)),
    layers.Dropout(0.7),
    layers.Dense(15, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Checkpoint and Early Stopping
checkpoint = ModelCheckpoint('best_model_upgraded.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

model.fit(X_train, y_train, epochs=1000, batch_size=40, validation_data=(X_test, y_test), callbacks=[checkpoint, early_stopping])

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")

predictions = model.predict(X_test)
predicted_classes = (predictions > 0.5).astype("int32")
