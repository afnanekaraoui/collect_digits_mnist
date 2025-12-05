from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import io
import uuid
import numpy as np
from datetime import datetime
from supabase import create_client, Client
from dotenv import load_dotenv
import os
import zipfile
import base64

# Load .env
load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = Flask(__name__)
CORS(app)

BUCKET_NAME = "digits"   # make sure this bucket exists


# =======================================================
# 🚀 SAVE DIGIT (UPLOAD TO SUPABASE STORAGE)
# =======================================================
@app.route('/save_digit', methods=['POST'])
def save_digit():
    try:
        if 'image' not in request.files or 'label' not in request.form:
            return jsonify({'error': 'Image and label required'}), 400

        image_file = request.files['image']
        label = request.form['label']

        if not label.isdigit() or int(label) not in range(10):
            return jsonify({'error': 'Label must be between 0–9'}), 400

        # Process image
        img = Image.open(image_file).convert('L')
        if img.size != (28, 28):
            img = img.resize((28, 28), Image.BILINEAR)

        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        file_bytes = img_bytes.getvalue()

        # Unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        filename = f"digit_{label}_{timestamp}_{unique_id}.png"

        # Upload path → "0/xxx", "1/xxx"
        upload_path = f"{label}/{filename}"

        # Upload to Supabase
        response = supabase.storage.from_(BUCKET_NAME).upload(
            upload_path,
            file_bytes,
            file_options={"content-type": "image/png"}
        )

        # --- FIXED ERROR CHECK ---
        if response.error:
            return jsonify({"error": str(response.error)}), 500

        return jsonify({
            "success": True,
            "filename": filename,
            "path": upload_path
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =======================================================
# 📊 GET STATISTICS (COUNT FILES IN SUPABASE STORAGE)
# =======================================================
@app.route("/stats", methods=["GET"])
def stats():
    try:
        stats = {}
        total = 0

        for label in range(10):
            folder = f"{label}"
            files = supabase.storage.from_(BUCKET_NAME).list(folder)

            count = len(files)
            stats[str(label)] = count
            total += count

        return jsonify({
            "stats": stats,
            "total": total
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



# =======================================================
# 📁 LIST ALL FILES FROM SUPABASE
# =======================================================
@app.route("/list_files", methods=["GET"])
def list_files():
    try:
        result = {}

        for label in range(10):
            folder = f"{label}"
            files = supabase.storage.from_(BUCKET_NAME).list(folder)
            result[str(label)] = [f["name"] for f in files]

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500



# =======================================================
# 🧩 DOWNLOAD ALL IMAGES AS ZIP (DIRECTLY FROM SUPABASE)
# =======================================================
@app.route("/download_zip")
def download_zip():
    try:
        memory_file = io.BytesIO()

        with zipfile.ZipFile(memory_file, "w", zipfile.ZIP_DEFLATED) as zf:
            for label in range(10):
                folder = f"{label}"
                files = supabase.storage.from_(BUCKET_NAME).list(folder)

                for file in files:
                    file_path = f"{folder}/{file['name']}"
                    file_bytes = supabase.storage.from_(BUCKET_NAME).download(file_path)

                    zf.writestr(file_path, file_bytes)

        memory_file.seek(0)
        return send_file(
            memory_file,
            mimetype="application/zip",
            as_attachment=True,
            download_name="digits_dataset.zip"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500



# =======================================================
# 🔢 DOWNLOAD NUMPY (X, y) FROM SUPABASE STORAGE
# =======================================================
@app.route("/download_numpy")
def download_numpy():
    try:
        X = []
        y = []

        for label in range(10):
            folder = f"{label}"
            files = supabase.storage.from_(BUCKET_NAME).list(folder)

            for file in files:
                file_path = f"{folder}/{file['name']}"
                file_bytes = supabase.storage.from_(BUCKET_NAME).download(file_path)

                img = Image.open(io.BytesIO(file_bytes)).convert("L")
                X.append(np.array(img))
                y.append(label)

        X = np.array(X)
        y = np.array(y)

        mem = io.BytesIO()
        np.savez_compressed(mem, X=X, y=y)
        mem.seek(0)

        return send_file(
            mem,
            mimetype="application/octet-stream",
            as_attachment=True,
            download_name="digits_numpy.npz"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500



# =======================================================
# HEALTH CHECK
# =======================================================
@app.route("/health")
def health():
    return {"status": "running"}



# =======================================================
# ROOT
# =======================================================
@app.route("/")
def index():
    return "MNIST Digit Collector Backend with Supabase Storage"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)