from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load_model('models/cnn_model.keras')
print("Model loaded successfully.")
print(model.summary())  # Add this line

def prepare_image(image_path):
    # Load and preprocess the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return None
    print(f"Original image shape: {img.shape}")  # Add this line
    img = cv2.resize(img, (224, 224))
    print(f"Resized image shape: {img.shape}")  # Add this line
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    print(f"Final preprocessed image shape: {img.shape}")  # Add this line
    return img

# API route to accept image and return prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save the uploaded file temporarily
    file_path = os.path.join('static', file.filename)
    file.save(file_path)

    # Prepare the image
    image = prepare_image(file_path)
    if image is None:
        return jsonify({'error': 'Failed to process image'}), 500

    # Log the file path and input shape for debugging
    print(f"Processing image: {file_path}")
    print(f"Image shape after preprocessing: {image.shape}")

    # Make a prediction
    prediction = model.predict(image)
    print(f"Raw prediction output: {prediction}")

    # Interpret the prediction
    probability = prediction[0][0]
    print(f"Extracted probability: {probability}")
    result = 'FAKE' if probability > 0.8 else 'REAL'
    print(f"Final result: {result}")

    # Return the result and probability in the response
    return jsonify({'prediction': result, 'probability': float(probability)})

if __name__ == '__main__':
    # Ensure 'static' folder exists
    if not os.path.exists('static'):
        os.makedirs('static')

    app.run(debug=True)