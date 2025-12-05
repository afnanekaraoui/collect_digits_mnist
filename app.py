from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import os
import uuid
import zipfile
import io
import numpy as np
from datetime import datetime
from supabase import create_client, Client
from dotenv import load_dotenv
load_dotenv() 

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
app = Flask(__name__)
CORS(app)  # Enable requests from any domain

# Configuration
DATASET_ROOT = 'dataset'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def init_dataset_folders():
    """Create folders for each digit class (0-9)"""
    for digit in range(10):
        folder_path = os.path.join(DATASET_ROOT, str(digit))
        os.makedirs(folder_path, exist_ok=True)
    print("✅ Dataset folders initialized (0-9)")

@app.route('/save_digit', methods=['POST'])
def save_digit():
    try:
        if 'image' not in request.files or 'label' not in request.form:
            return jsonify({'error': 'Image and label required'}), 400

        image_file = request.files['image']
        label = request.form['label']

        if not label.isdigit() or int(label) not in range(10):
            return jsonify({'error': 'Label must be 0-9'}), 400

        # Process image with PIL
        img = Image.open(image_file).convert('L')
        if img.size != (28, 28):
            img = img.resize((28, 28), Image.BILINEAR)

        # Save to in-memory bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        # Unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        filename = f'digit_{label}_{timestamp}_{unique_id}.png'
        path_in_bucket = f"{label}/{filename}"

        # Upload to Supabase
        res = supabase.storage.from_('digits').upload(path_in_bucket, img_bytes)

        if res.get("error"):
            return jsonify({'error': res["error"]["message"]}), 500

        return jsonify({
            'success': True,
            'message': f'Saved digit {label} in Supabase Storage',
            'filename': filename,
            'label': label
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get dataset statistics"""
    try:
        stats = {}
        total = 0
        
        for digit in range(10):
            folder_path = os.path.join(DATASET_ROOT, str(digit))
            if os.path.exists(folder_path):
                count = len([f for f in os.listdir(folder_path) if f.endswith('.png')])
                stats[str(digit)] = count
                total += count
            else:
                stats[str(digit)] = 0
        
        return jsonify({
            'stats': stats,
            'total': total
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== DATA RETRIEVAL ENDPOINTS ====================

@app.route('/download_zip', methods=['GET'])
def download_zip():
    """
    Download entire dataset as ZIP file
    Usage: curl http://your-server.com/download_zip -o dataset.zip
    """
    try:
        memory_file = io.BytesIO()
        
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for digit in range(10):
                folder_path = os.path.join(DATASET_ROOT, str(digit))
                if os.path.exists(folder_path):
                    for filename in os.listdir(folder_path):
                        if filename.endswith('.png'):
                            file_path = os.path.join(folder_path, filename)
                            # Add to zip with folder structure
                            zf.write(file_path, os.path.join(str(digit), filename))
        
        memory_file.seek(0)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        zip_filename = f'mnist_dataset_{timestamp}.zip'
        
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=zip_filename
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_numpy', methods=['GET'])
def download_numpy():
    """
    Export dataset as NumPy arrays
    Returns: JSON with base64-encoded arrays
    """
    try:
        X = []
        y = []
        
        for label in range(10):
            folder_path = os.path.join(DATASET_ROOT, str(label))
            if os.path.exists(folder_path):
                for filename in os.listdir(folder_path):
                    if filename.endswith('.png'):
                        img_path = os.path.join(folder_path, filename)
                        img = Image.open(img_path).convert('L')
                        X.append(np.array(img))
                        y.append(label)
        
        if len(X) == 0:
            return jsonify({'error': 'No images found'}), 404
        
        X = np.array(X)
        y = np.array(y)
        
        # Create ZIP with NumPy files
        memory_file = io.BytesIO()
        np.savez_compressed(memory_file, X=X, y=y)
        memory_file.seek(0)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        return send_file(
            memory_file,
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name=f'mnist_numpy_{timestamp}.npz'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/list_files', methods=['GET'])
def list_files():
    """List all files in dataset"""
    try:
        files = {}
        
        for label in range(10):
            folder_path = os.path.join(DATASET_ROOT, str(label))
            if os.path.exists(folder_path):
                files[str(label)] = [f for f in os.listdir(folder_path) if f.endswith('.png')]
            else:
                files[str(label)] = []
        
        total = sum(len(f) for f in files.values())
        
        return jsonify({
            'files': files,
            'total': total
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'message': 'MNIST Digit Collector Backend is active'
    }), 200

@app.route('/')
def index():
    return "MNIST Digit Collector Backend is running! Visit /stats or /save_digit"

if __name__ == '__main__':
    print("=" * 50)
    print("🚀 MNIST Digit Collector Backend")
    print("=" * 50)
    
    init_dataset_folders()
    
    print(f"📁 Dataset root: {os.path.abspath(DATASET_ROOT)}")
    print("🌐 Server starting...")
    print("=" * 50)
    
    # For production deployment
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=False, host='0.0.0.0', port=port) 