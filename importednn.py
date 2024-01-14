from sklearn.datasets import load_breast_cancer
import tensorflow as tf
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
X = data.data
y = data.target

# Load the model

best_model = load_model('best_model_basic.h5')

# Data Preprocessing: Scale the features using the same scaler from training. This is necessary for testing the upgraded model.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Uncomment the line below to test the upgraded model
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

loss, accuracy = best_model.evaluate(X_test, y_test)

print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")

predictions = best_model.predict(X_test)
