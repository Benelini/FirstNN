from sklearn.datasets import load_breast_cancer
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.models import load_model

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    layers.Dense(30, activation="elu", input_shape=(X_train.shape[1],)),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], )
checkpoint = ModelCheckpoint('best_model_basic.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min',)
model.fit(X_train, y_train, epochs=1000, batch_size=40, validation_data=(X_test, y_test), callbacks=[checkpoint])

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")

predictions = model.predict(X_test)
predicted_classes = (predictions > 0.5).astype("int32")
