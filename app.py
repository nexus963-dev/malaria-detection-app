import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model
MODEL = None

def load_model():
    global MODEL
    if MODEL is None:
        try:
            MODEL = tf.keras.models.load_model('malaria_cnn_complete_model.keras')
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e
    return MODEL

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not read image file")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = image.astype('float32') / 255.0
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
    filepath = None
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        processed_image = preprocess_image(filepath)
        if processed_image is None:
            return jsonify({'error': 'Error processing image'}), 500
        
        model = load_model()
        prediction = model.predict(processed_image, verbose=0)
        probability = float(prediction[0][0])
        
        if probability > 0.5:
            result = "Uninfected"
            confidence = probability * 100
        else:
            result = "Parasitized"
            confidence = (1 - probability) * 100
        
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({
            'prediction': result,
            'confidence': round(confidence, 2),
            'probability': round(probability, 4),
            'status': 'success'
        })
        
    except Exception as e:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL is not None,
        'tensorflow_version': tf.__version__
    })

if __name__ == '__main__':
    print("🚀 Starting Malaria Detection App...")
    load_model()
    
    # Use environment port for cloud deployment
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
