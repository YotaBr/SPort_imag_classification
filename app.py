from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from werkzeug.utils import secure_filename
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from supabase import create_client, Client
from datetime import datetime, timezone
import logging
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
import requests
import json
from dotenv import load_dotenv

# 🔐 Load environment variables
load_dotenv()

# ⚙️ Flask + Logging Setup
app = Flask(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# 🔑 Load Secrets from Environment
MODEL_DRIVE_FILE_ID = os.getenv("MODEL_DRIVE_FILE_ID")
SERVICE_ACCOUNT_JSON = json.loads(os.getenv("SERVICE_ACCOUNT_JSON"))
DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SCOPES = ['https://www.googleapis.com/auth/drive.file']
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
MODEL_PATH = "model.h5"

# ⬇️ Safe model download from Google Drive
def download_model(file_id, dest_path):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    logging.info("📥 Attempting model download from Google Drive...")

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if token:
        response = session.get(URL, params={'id': file_id, 'confirm': token}, stream=True)

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

    logging.info("✅ Model downloaded and saved to %s", dest_path)

    if os.path.getsize(dest_path) < 1_000_000:
        raise RuntimeError("❌ Model file too small. Likely corrupted or invalid download.")

# 🧼 Clean model if invalid or corrupted
def ensure_model_valid(file_path, file_id):
    if os.path.exists(file_path):
        if os.path.getsize(file_path) < 1_000_000:
            logging.warning("⚠️ Model file is too small, likely corrupted. Re-downloading.")
            os.remove(file_path)
    if not os.path.exists(file_path):
        download_model(file_id, file_path)

# 🧠 Prepare model
ensure_model_valid(MODEL_PATH, MODEL_DRIVE_FILE_ID)
model = tf.keras.models.load_model(MODEL_PATH)
target_size = model.input_shape[1:3]

# 🏷️ Class labels
CLASS_NAMES = {
    0: 'Basketball',
    1: 'Cricket',
    2: 'Rugby',
    3: 'badminton',
    4: 'boxing',
    5: 'football',
    6: 'swimming',
    7: 'wrestling'
}

# 🧼 Image preprocessing
def preprocess_image(file_path):
    img = Image.open(file_path).convert('RGB')
    img = img.resize(target_size)
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

# ☁️ Google Drive upload setup
def get_drive_service():
    creds = service_account.Credentials.from_service_account_info(SERVICE_ACCOUNT_JSON, scopes=SCOPES)
    return build('drive', 'v3', credentials=creds)

def upload_to_drive(filepath):
    service = get_drive_service()
    file_metadata = {'name': os.path.basename(filepath), 'parents': [DRIVE_FOLDER_ID]}
    media = MediaFileUpload(filepath, mimetype='image/jpeg')
    uploaded = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    return uploaded.get('id')

# 🧬 Supabase Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 🌐 Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'status': 'fail', 'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '' or not file.content_type.startswith('image/'):
        return jsonify({'status': 'fail', 'error': 'Invalid file'}), 400

    file.seek(0, os.SEEK_END)
    if file.tell() > MAX_FILE_SIZE:
        return jsonify({'status': 'fail', 'error': 'File too large'}), 400
    file.seek(0)

    filename = secure_filename(file.filename)
    temp_path = f"temp_{datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}"

    try:
        file.save(temp_path)
        drive_file_id = upload_to_drive(temp_path)

        img = preprocess_image(temp_path)
        preds = model.predict(img)
        pred_index = int(np.argmax(preds[0]))
        confidence = float(preds[0][pred_index])
        result_class = CLASS_NAMES.get(pred_index, "Unknown")

        supabase.table('predictions').insert({
            'class': result_class,
            'confidence': confidence,
            'drive_file_id': drive_file_id,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }).execute()

        return jsonify({'status': 'success', 'class': result_class, 'confidence': confidence})
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'status': 'fail', 'error': 'Prediction failed'}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# 🚀 Launch
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
