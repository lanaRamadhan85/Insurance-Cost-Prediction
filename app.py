"""
Flask Web Application untuk Prediksi Biaya Asuransi
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model dan encoders
MODEL_PATH = 'model/insurance_model.pkl'
ENCODERS_PATH = 'model/encoders.pkl'

def load_model():
    """Load trained model dan encoders"""
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(ENCODERS_PATH, 'rb') as f:
            encoders = pickle.load(f)
        return model, encoders
    except FileNotFoundError:
        return None, None

model, encoders = load_model()

@app.route('/')
def index():
    """Render halaman utama"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint untuk prediksi biaya asuransi"""
    
    if model is None or encoders is None:
        return jsonify({
            'error': 'Model tidak ditemukan. Silakan jalankan train_model.py terlebih dahulu.'
        }), 500
    
    try:
        # Ambil data dari form
        age = int(request.form.get('age'))
        sex = request.form.get('sex')
        bmi = float(request.form.get('bmi'))
        children = int(request.form.get('children'))
        smoker = request.form.get('smoker')
        region = request.form.get('region')
        
        # Validasi input
        if age < 18 or age > 100:
            return jsonify({'error': 'Umur harus antara 18-100 tahun'}), 400
        if bmi < 10 or bmi > 60:
            return jsonify({'error': 'BMI harus antara 10-60'}), 400
        if children < 0 or children > 10:
            return jsonify({'error': 'Jumlah anak harus antara 0-10'}), 400
        
        # Encoding variabel kategorikal
        sex_encoded = encoders['sex'].transform([sex])[0]
        smoker_encoded = encoders['smoker'].transform([smoker])[0]
        region_encoded = encoders['region'].transform([region])[0]
        
        # Membuat array untuk prediksi
        features = np.array([[age, bmi, children, sex_encoded, smoker_encoded, region_encoded]])
        
        # Prediksi
        prediction = model.predict(features)[0]
        
        # Format hasil
        result = {
            'prediction': round(float(prediction), 2),
            'formatted_prediction': f"${prediction:,.2f}"
        }
        
        return jsonify(result)
    
    except ValueError as e:
        return jsonify({'error': f'Input tidak valid: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Terjadi kesalahan: {str(e)}'}), 500

@app.route('/img/<path:filename>')
def serve_img(filename):
    """Serve images from img folder"""
    return send_from_directory('img', filename)

@app.route('/health')
def health():
    """Health check endpoint"""
    if model is None or encoders is None:
        return jsonify({'status': 'unhealthy', 'message': 'Model tidak ditemukan'}), 503
    return jsonify({'status': 'healthy', 'message': 'Model siap digunakan'})

if __name__ == '__main__':
    # Cek apakah model sudah ada
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODERS_PATH):
        print("="*60)
        print("WARNING: Model tidak ditemukan!")
        print("="*60)
        print("Silakan jalankan train_model.py terlebih dahulu untuk melatih model.")
        print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)

