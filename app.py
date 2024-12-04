import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load precomputed features and model
features = np.load('features.npy')  # Precomputed image features
image_paths = np.load('image_paths.npy', allow_pickle=True)  # Corresponding image paths
model = load_model('style_matching_model.h5')  # Your trained model

# Folder to save uploaded images temporarily
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create the folder if it doesn't exist
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Utility function to preprocess the uploaded image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Adjust size based on model input
    img = img_to_array(img)
    img = img / 255.0  # Normalize the image
    img = img.reshape((1, 224, 224, 3))  # Reshape for the model
    return img

# Home route: Serve the HTML page
@app.route('/')
def home():
    return render_template('index.html')  # Serve the index.html file for the frontend

# Route to handle image uploads and return matching styles
@app.route('/match', methods=['POST'])
def match_style():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded image temporarily to process it
    uploaded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(uploaded_image_path)

    try:
        # Preprocess the uploaded image
        image = preprocess_image(uploaded_image_path)

        # Get the feature vector of the uploaded image from the model
        feature_vector = model.predict(image)  # If the model outputs a feature vector directly

        # Compute cosine similarity with precomputed features
        similarity_scores = cosine_similarity(feature_vector, features)

        # Get the indices of the top 5 most similar images
        top_indices = np.argsort(similarity_scores[0])[::-1][:5]

        # Collect the paths of the top matching images
        matches = [
            {
                'path': f"/static/{image_paths[idx]}",  # Ensure accessible static path
                'score': float(similarity_scores[0][idx])  # Ensure score is JSON serializable
            }
            for idx in top_indices
        ]

        return jsonify({'matches': matches})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up uploaded file after processing
        if os.path.exists(uploaded_image_path):
            os.remove(uploaded_image_path)

if __name__ == '__main__':
    app.run(debug=True)
