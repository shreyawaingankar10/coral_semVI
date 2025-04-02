import os
from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("coral_classifier.h5")

app = Flask(__name__)

# Define upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the folder exists
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Check class indices to ensure correct label mapping
class_indices = {'Bleached': 0, 'Healthy': 1}  # Modify if necessary
class_labels = {v: k for k, v in class_indices.items()}  # Reverse mapping

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the image
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Process and predict
    result = process_and_predict(filepath)
    return jsonify({'result': result, 'image_path': filepath})

def process_and_predict(img_path):
    """Preprocess image and make prediction"""
    # Load image and convert to RGB
    img = image.load_img(img_path, target_size=(224, 224), color_mode="rgb")
    
    # Convert to numpy array and normalize
    img_array = image.img_to_array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img_array)[0][0]

    # Debugging logs
    print(f"Processed Image Shape: {img_array.shape}")
    print(f"Raw Model Prediction: {prediction}")

    # Convert prediction to class label
    return "Bleached" if prediction < 0.5 else "Healthy"

if __name__ == "__main__":
    app.run(debug=True)
