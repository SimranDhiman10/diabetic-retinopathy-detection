from flask import Flask, render_template, request
import cv2
import numpy as np
import tensorflow as tf
import os
import time
from datetime import datetime, timedelta
from threading import Thread
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Ensure upload folder exists
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained CNN model
MODEL_PATH = 'model/cnn_retinopathy_model.keras'
print("Loading model... Please wait.")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Label mapping for predictions
label_names = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

# Set a time threshold (e.g., delete files after 10 minutes)
FILE_EXPIRATION_TIME = timedelta(minutes=10)

# Create a dictionary to store file paths and upload time
uploaded_files = {}

# Function to delete files older than the threshold
def cleanup_old_files():
    while True:
        current_time = datetime.now()
        for file_path, upload_time in list(uploaded_files.items()):
            if current_time - upload_time > FILE_EXPIRATION_TIME:
                try:
                    os.remove(file_path)
                    print(f"Deleted old file: {file_path}")
                    del uploaded_files[file_path]
                except FileNotFoundError:
                    pass  # In case the file was already deleted
        time.sleep(60)  # Run every minute to check for expired files

# Start the cleanup thread
cleanup_thread = Thread(target=cleanup_old_files, daemon=True)
cleanup_thread.start()

# Function to preprocess the input image for the CNN
def preprocess_image(file_path, img_size=(224, 224)):
    img = cv2.imread(file_path)
    if img is None:
        return None
    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model not loaded. Please check the server logs.", 500

    if 'file' not in request.files:
        return "No file part in the request", 400

    file = request.files['file']

    if not file or not file.filename:
        return "No file selected", 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    uploaded_files[file_path] = datetime.now()

    # Preprocess the image for the CNN
    preprocessed_image = preprocess_image(file_path)
    if preprocessed_image is None:
        return "Could not read the uploaded image.", 400

    # Make prediction
    prediction = model.predict(preprocessed_image)[0]
    predicted_class = np.argmax(prediction).item()
    confidence = float(np.max(prediction)) * 100
    predicted_label = label_names[predicted_class]

    relative_image_path = f"uploads/{filename}"
    return render_template(
        'result.html',
        image_path=relative_image_path,
        prediction=predicted_label,
        predicted_class=predicted_class,
        confidence=f"{confidence:.2f}"
    )

if __name__ == '__main__':
    app.run(debug=True, port=8000)


