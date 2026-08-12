# CADIF: Correspondence Analysis Decision Intelligence Framework

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

## Deskripsi Proyek
**CADIF (Correspondence Analysis Decision Intelligence Framework)** adalah sebuah kerangka sistem pendukung keputusan *data-driven* yang mengintegrasikan eksplorasi multivariat dengan mesin keputusan multikriteria. Aplikasi ini dibangun menggunakan kerangka kerja Streamlit berbasis Python.

Saat ini, dasbor dikonfigurasi menggunakan **Studi Kasus Penderita Serangan Jantung**. Namun, karena sifat algoritma pemodelan matematika di dalamnya yang dinamis, kode pada sistem ini **sangat fleksibel dan dapat diedit untuk memproses dataset dari domain lain** (misalnya bisnis, sosial, atau lingkungan).

### Sumber Data
Dataset bawaan yang digunakan pada sistem ini berasal dari indikator kesehatan CDC (BRFSS 2022) yang dipublikasikan di Kaggle: 
[Personal Key Indicators of Heart Disease](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease/data)

## Arsitektur & Fitur Utama
Aplikasi dasbor ini dibagi menjadi 5 modul utama yang beroperasi secara berurutan:

*   **1. Data Management:** Mengelola input data mentah (CSV) atau tabel kontingensi. Modul ini melakukan binorisasi otomatis pada variabel numerik (seperti BMI, jam tidur, hari kesehatan fisik/mental) berdasarkan standar WHO dan CDC.
*   **2. SCA & MCA Engine:** Mengekstraksi struktur ruang geometri menggunakan Analisis Korespondensi Sederhana (Bivariat) dan Analisis Korespondensi Berganda (Multivariat). Pada mode MCA, sistem menggunakan Matriks Burt dan mengaplikasikan koreksi Benzécri untuk perhitungan inersia yang akurat.
*   **3. Risk Intelligence Engine:** Membobotkan kriteria spasial (*Contribution, Cos2, Projection to Target*) secara objektif menggunakan *Entropy Weight Method* (EWM). Hasilnya kemudian diagregasi menggunakan algoritma TOPSIS untuk menghasilkan CADIF Score (skala 0-10).
*   **4. Decision Engine:** Melakukan stratifikasi dan klasifikasi skor risiko menjadi tingkat prioritas (Low, Medium, High, Critical). Batas pemotongan dihitung secara dinamis menggunakan batas Kuartil (Q1, Q2, Q3) dari distribusi data yang dimasukkan, dilengkapi dengan *override* otomatis untuk faktor berasosiasi negatif.
*   **5. Executive Summary:** Mengompilasi hasil matematis ke dalam visualisasi *Pie Chart* dan men-*generate* tabel narasi interpretasi secara otomatis sebagai panduan intervensi klinis (seperti *Immediate Intervention* atau *Targeted Prevention*).

## Cara Menjalankan Aplikasi di Komputer Lokal

Jika Anda ingin menjalankan atau memodifikasi aplikasi ini di komputer Anda sendiri, ikuti panduan berikut:

1. **Clone repositori ini:**
   ```bash
   git clone [https://github.com/adhityorafael/cadif-dashboard.git](https://github.com/adhityorafael/cadif-dashboard.git)
   cd cadif-dashboard

2. **Install library yang dibutuhkan**
   Pastikan Anda memiliki file requirements.txt yang memuat library dasar (seperti streamlit, pandas, numpy, plotly), lalu jalankan:
   
   
3. **Jalankan aplikasi Streamlit**
