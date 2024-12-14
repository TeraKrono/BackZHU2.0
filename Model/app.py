# app.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Lambda
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from mtcnn import MTCNN
import cv2
from flask import Flask, request, jsonify
import base64

# Initialize Flask app
app = Flask(__name__)

# Initialize MTCNN face detector
detector = MTCNN()

def create_embedding_model(input_shape=(224, 224, 3)):
    # Load ResNet50 as backbone
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    
    # Freeze base_model layers to use it as a pure feature extractor
    for layer in base_model.layers:
        layer.trainable = False
    
    x = base_model.output
    # Global average pooling to get a compact feature vector
    x = GlobalAveragePooling2D()(x)
    # Produce a 128-dimensional embedding
    x = Dense(128)(x)
    # L2-normalize the embeddings
    x = Lambda(lambda v: tf.math.l2_normalize(v, axis=1))(x)
    
    model = Model(inputs=base_model.input, outputs=x)
    return model

# Path to the saved embedding model weights
model_path = 'test.weights.h5'  

input_shape = (224, 224, 3)
embedding_model = create_embedding_model(input_shape)
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model weights not found at {model_path}")
embedding_model.load_weights(model_path)
print("Model loaded successfully.")

def compute_embedding(image):
    # Convert image to RGB if it's BGR (OpenCV uses BGR)
    if image.shape[-1] == 3:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = image
    # Detect faces
    results = detector.detect_faces(img_rgb)
    if results:
        # Find the largest face
        max_area = 0
        largest_face = None
        for result in results:
            x, y, width, height = result['box']
            area = width * height
            if area > max_area:
                max_area = area
                largest_face = result['box']
        if largest_face is not None:
            x, y, width, height = largest_face
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = x1 + width
            y2 = y1 + height
            face = img_rgb[y1:y2, x1:x2]
            # Resize face to 224x224 for ResNet
            face = cv2.resize(face, (224, 224)).astype('float32')
            # Preprocess using preprocess_input
            face = np.expand_dims(face, axis=0)
            face = preprocess_input(face)
            # Get embedding
            embedding = embedding_model.predict(face)
            return embedding[0]
    else:
        print("No face detected in the image.")
        return None

# Endpoint to compute embedding from a base64-encoded image
@app.route('/compute_embedding', methods=['POST'])
def compute_embedding_endpoint():
    if not request.is_json:
        return jsonify({'error': 'Request must be in JSON format'}), 400
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400
    image_base64 = data['image']
    try:
        # Decode the base64 string
        img_bytes = base64.b64decode(image_base64)
        # Convert bytes to NumPy array
        npimg = np.frombuffer(img_bytes, np.uint8)
        # Decode image
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({'error': f'Invalid image data: {str(e)}'}), 400

    embedding = compute_embedding(image)
    if embedding is not None:
        embedding_list = embedding.tolist()
        return jsonify({'embedding': embedding_list}), 200
    else:
        return jsonify({'error': 'Face not detected in the image'}), 400

# Run the app
if __name__ == '__main__':
    print("Starting Flask app...")
    # Set debug=False in production
    app.run(host='0.0.0.0', port=5000, debug=False)
    print("Flask app has stopped.")
