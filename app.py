from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
from feature_extraction import extract_features
import joblib
import base64
from io import BytesIO
import matplotlib.pyplot as plt

app = Flask(__name__)

# Konfigurasi path
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Pastikan folder upload ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model dan scaler
try:
    model = joblib.load('model/model_svm.pkl')
    scaler = joblib.load('model/scaler.pkl')
    print("✅ Model dan scaler berhasil dimuat!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    scaler = None

# Fungsi untuk validasi ekstensi file
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

def image_to_base64(image):
    """Konversi gambar OpenCV ke base64"""
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

def plot_to_base64():
    """Konversi plot matplotlib ke base64"""
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=80)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def extract_visual_features(image):
    """Ekstrak fitur visual untuk ditampilkan di halaman hasil"""
    features = {}
    
    # 1. Edge Detection (Canny)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    features['edge'] = image_to_base64(edges_rgb)
    
    # 2. Histogram Warna
    plt.figure(figsize=(6, 4))
    colors = ('b', 'g', 'r')
    channel_names = ('Biru', 'Hijau', 'Merah')
    for i, (color, name) in enumerate(zip(colors, channel_names)):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color, label=name)
    plt.title('Distribusi Warna')
    plt.xlabel('Intensitas Warna')
    plt.ylabel('Frekuensi')
    plt.xlim([0, 256])
    plt.grid(True, alpha=0.3)
    plt.legend()
    features['histogram'] = plot_to_base64()
    
    # 3. Tekstur (Grayscale)
    # Tingkatkan kontras untuk visualisasi yang lebih baik
    gray_eq = cv2.equalizeHist(gray)
    texture = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2RGB)
    features['texture'] = image_to_base64(texture)
    
    # 4. Gambar Asli (resize)
    resized_img = cv2.resize(image, (300, 300))
    features['original'] = image_to_base64(resized_img)
    
    # 5. Bentuk (Contour)
    # Temukan kontur pada gambar tepi
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = np.zeros_like(image)
    # Gambar kontur terbesar
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(contour_img, [largest_contour], -1, (0, 255, 0), 2)
    features['contour'] = image_to_base64(contour_img)
    
    return features

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Cek jika file ada dalam request
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    # Cek jika user tidak memilih file
    if file.filename == '':
        return redirect(request.url)
    
    # Cek jika file valid
    if file and allowed_file(file.filename):
        # Buat nama file yang aman
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Simpan file
        file.save(filepath)
        print(f"💾 File disimpan di: {filepath}")
        
        # Baca gambar menggunakan OpenCV
        try:
            image = cv2.imread(filepath)
            if image is None:
                raise ValueError("Gagal membaca gambar")
                
            # Ekstrak fitur untuk klasifikasi
            features = extract_features(image)
            
            # Normalisasi fitur
            if scaler:
                features_scaled = scaler.transform([features])
            else:
                return "Error: Scaler tidak tersedia", 500
            
            # Prediksi
            if model:
                prediction = model.predict(features_scaled)[0]
                probability = model.predict_proba(features_scaled)[0]
            else:
                return "Error: Model tidak tersedia", 500
            
            # Format hasil probabilitas
            probabilities = {
                'Dara': f"{probability[0]*100:.2f}%",
                'Ijo': f"{probability[1]*100:.2f}%",
                'Simping': f"{probability[2]*100:.2f}%"
            }
            
            # Ekstrak fitur visual untuk ditampilkan
            visual_features = extract_visual_features(image)
            
            # Path relatif untuk template
            rel_path = os.path.join('uploads', filename).replace('\\', '/')
            print(f"📍 Rel path: {rel_path}")
            
            return render_template('result.html', 
                                   image_path=rel_path,
                                   prediction=prediction,
                                   probabilities=probabilities,
                                   visual_features=visual_features)
            
        except Exception as e:
            print(f"❌ Error processing image: {e}")
            return render_template('error.html', error=str(e))
    
    return "Format file tidak didukung. Harap unggah file gambar (PNG, JPG, JPEG, GIF)"

@app.route('/model-evaluation')
def model_evaluation():
    try:
        # Load metrik evaluasi
        metrics = joblib.load('model/evaluation_metrics.pkl')
        
        return render_template('evaluation.html', 
                               accuracy=metrics['accuracy'],
                               report=metrics['report'],
                               conf_matrix=metrics['confusion_matrix'])
    except Exception as e:
        print(f"Error loading evaluation metrics: {e}")
        return render_template('error.html', error="Evaluasi model tidak tersedia")

@app.errorhandler(413)
def request_entity_too_large(error):
    return "Ukuran file terlalu besar (maksimal 16MB)", 413

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
    