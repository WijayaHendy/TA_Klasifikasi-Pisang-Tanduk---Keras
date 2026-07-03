from flask import Flask, request, render_template
from flask_sqlalchemy import SQLAlchemy
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from datetime import datetime

# --- 1. INISIALISASI APLIKASI ---
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///klasifikasi.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Inisialisasi Database
db = SQLAlchemy(app)

# --- 2. MODEL DATABASE ---
class HasilKlasifikasi(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100))
    hasil = db.Column(db.String(50))
    waktu = db.Column(db.DateTime, default=datetime.utcnow)

# --- 3. KONFIGURASI MODEL CNN ---
MODEL_PATH = 'Citra Pisang Tanduk_Xception-Pisang Tanduk Xception-93.06.keras'
try:
    model = load_model(MODEL_PATH)
    print(f"Berhasil memuat model: {MODEL_PATH}")
except Exception as e:
    print(f"Gagal memuat model: {e}")
    model = None

# Daftar Kelas Pisang
classes = ['Pisang Mentah', 'Pisang Setengah Matang', 'Pisang Matang', 'Pisang Terlalu Matang', 'Pisang Busuk']

# --- 4. ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return "Model tidak ditemukan! Pastikan file .keras ada di folder proyek.", 500
    
    file = request.files.get('file')
    if not file or file.filename == '':
        return 'Tidak ada file yang dipilih', 400

    upload_folder = os.path.join('static', 'uploads')
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    
    filepath = os.path.join(upload_folder, file.filename)
    file.save(filepath)

    # Pra-pemrosesan Citra (Target size 256x256 sesuai model Xception)
    img = image.load_img(filepath, target_size=(256, 256))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence_val = float(np.max(prediction)) * 100

    # --- LOGIKA FILTER "BUKAN PISANG" ---
    # Jika akurasi sangat tinggi pada objek salah (seperti foto manusia), 
    # Menggunakan threshold yang sangat ketat (99.8%)
    if confidence_val < 99.8: 
        result = 'Bukan Pisang Tanduk'
    else:
        result = classes[class_index]

    # Simpan ke Database Riwayat
    riwayat = HasilKlasifikasi(filename=file.filename, hasil=result)
    db.session.add(riwayat)
    db.session.commit()

    # Kirim hasil ke result.html
    return render_template('result.html', 
                           prediction=result, 
                           image_path=file.filename, 
                           confidence=f"{confidence_val:.2f}")

@app.route('/riwayat')
def riwayat():
    data = HasilKlasifikasi.query.order_by(HasilKlasifikasi.waktu.desc()).all()
    return render_template('riwayat.html', data=data)

@app.route('/evaluasi')
def evaluasi():
    # Masukkan angka yang Anda dapatkan dari Langkah 1 tadi ke sini
    data_metrics = {
        'accuracy': '93.00',
        'precision': '94.00',
        'recall': '93.00',
        'f1_score': '93.00'
    }
    return render_template('evaluasi.html', metrics=data_metrics)

# --- 5. MENJALANKAN SERVER ---
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)