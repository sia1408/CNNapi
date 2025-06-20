from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

app = Flask(__name__)

model = load_model('models/cnn_model.keras')
print("Model loaded successfully.")
print(model.summary())

def prepare_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return None
    print(f"Original image shape: {img.shape}")
    img = cv2.resize(img, (224, 224))
    print(f"Resized image shape: {img.shape}")
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    print(f"Final preprocessed image shape: {img.shape}")
    return img

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    file_path = os.path.join('static', file.filename)
    file.save(file_path)

    image = prepare_image(file_path)
    if image is None:
        return jsonify({'error': 'Failed to process image'}), 500

    print(f"Processing image: {file_path}")
    print(f"Image shape after preprocessing: {image.shape}")

    prediction = model.predict(image)
    print(f"Raw prediction output: {prediction}")

    probability = prediction[0][0]
    print(f"Extracted probability: {probability}")
    result = 'FAKE' if probability > 0.5 else 'REAL'
    print(f"Final result: {result}")

    return jsonify({'prediction': result, 'probability': float(probability)})

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')

    app.run(debug=True)