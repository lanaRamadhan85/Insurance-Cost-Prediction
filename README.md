 INSURANCE COST PREDICTION - MACHINE LEARNING WEB APPLICATION

Aplikasi web berbasis Flask untuk memprediksi biaya asuransi kesehatan menggunakan model Machine Learning (Linear Regression). Project ini mencakup Exploratory Data Analysis (EDA), preprocessing data, model training, evaluasi model, dan web deployment.

====================================================================
OVERVIEW
========

Project ini bertujuan untuk:

1. Memprediksi biaya asuransi kesehatan berdasarkan fitur:

   * age
   * sex
   * bmi
   * children
   * smoker
   * region

2. Menganalisis faktor-faktor yang paling berpengaruh terhadap biaya asuransi.

3. Mengimplementasikan model dalam bentuk web application menggunakan Flask.

====================================================================
DATASET INFORMATION
===================

Nama file: insurance.csv
Jumlah data: 1338
Missing values: 0

Features:

* age        : Umur pemegang polis (18–64 tahun)
* sex        : Jenis kelamin (male/female)
* bmi        : Body Mass Index
* children   : Jumlah anak yang ditanggung
* smoker     : Status perokok (yes/no)
* region     : Wilayah (northeast, northwest, southeast, southwest)
* charges    : Biaya asuransi (target variable)

====================================================================
PROJECT STRUCTURE
=================


insurance-cost-prediction/

├── data/
│   └── insurance.csv
│
├── notebooks/
│   ├── project_data_mining_asuransi.ipynb
│   └── project_asuransi.ipynb
│
├── model/
│   ├── insurance_model.pkl
│   └── encoders.pkl
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── app.py
├── train_model.py
├── requirements.txt
└── README.md

====================================================================
INSTALLATION
============

1. Install dependencies:
   pip install -r requirements.txt

2. Train model terlebih dahulu:
   python train_model.py

   Proses ini akan:

   * Load dataset
   * Encode categorical features
   * Train Linear Regression model
   * Evaluasi model
   * Save model dan encoder ke folder model/

3. Jalankan aplikasi:
   python app.py

4. Buka browser dan akses:
   [http://localhost:5000](http://localhost:5000)

====================================================================
MODEL EVALUATION (TEST SET)
===========================

R2 Score : 0.7833
RMSE     : 5799.59
MAE      : 4186.51

Interpretasi:
Model mampu menjelaskan sekitar 78% variasi biaya asuransi.
Rata-rata error prediksi sekitar 4.000 – 5.800 dalam satuan biaya.

====================================================================
TECHNOLOGIES USED
=================

* Python 3.8+
* Flask
* Scikit-learn
* Pandas
* NumPy
* Jupyter Notebook
* HTML & CSS

====================================================================
FUTURE IMPROVEMENTS
===================

* Implementasi model alternatif (Random Forest, Gradient Boosting)
* Feature engineering lanjutan
* Deployment ke cloud (AWS / GCP / Heroku)
* Visualisasi prediksi tambahan
* Model versioning

====================================================================
AUTHOR
======

Lana Angger Ramadhan

====================================================================
LICENSE
=======

Project ini dibuat untuk tujuan pembelajaran dan portfolio.
