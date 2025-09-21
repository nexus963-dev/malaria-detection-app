from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model once when the app starts
MODEL = None

def load_model():
    global MODEL
    if MODEL is None:
        # Updated to load .keras format
        MODEL = tf.keras.models.load_model('malaria_cnn_complete_model.keras')
        print("Model loaded successfully!")
    return MODEL

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    try:
        # Read and resize image (same as training preprocessing)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        
        # Normalize pixel values to [0, 1] range
        image = image.astype('float32') / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, JPEG, or GIF'}), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess image
        processed_image = preprocess_image(filepath)
        if processed_image is None:
            return jsonify({'error': 'Error processing image'}), 500
        
        # Load model and make prediction
        model = load_model()
        prediction = model.predict(processed_image, verbose=0)  # verbose=0 to suppress output
        probability = float(prediction[0][0])
        
        # Interpret prediction (same logic as training)
        if probability > 0.5:
            result = "Uninfected"
            confidence = probability * 100
        else:
            result = "Parasitized"
            confidence = (1 - probability) * 100
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify({
            'prediction': result,
            'confidence': round(confidence, 2),
            'probability': round(probability, 4),
            'status': 'success'
        })
        
    except Exception as e:
        # Clean up file if it exists
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy', 
        'model_loaded': MODEL is not None,
        'model_format': 'keras'
    })

if __name__ == '__main__':
    print("Starting Malaria Detection Flask App...")
    print("Loading TensorFlow model...")
    
    # Load model on startup
    load_model()
    print("Flask app ready!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
