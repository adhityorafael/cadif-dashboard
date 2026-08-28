"""
Modul 3: Risk Intelligence Engine (TOPSIS & EWM)

Kontrak: Menerima DataFrame hasil SCA/MCA Engine (kolom Contribution_Gabungan_%,
Cos2_Gabungan, Projection_to_Target), mengembalikan DataFrame yang sama ditambah
kolom CADIF_Score (skala 0-10) dan Arah_Risiko.

Metodologi:
- Entropy Weight Method / EWM (Shannon, 1948; Zeleny, 1982) untuk bobot
  kriteria objektif berbasis dispersi data - kriteria dengan variansi lebih
  besar antar risk factor dianggap lebih informatif/membedakan, sehingga
  diberi bobot lebih besar.
- TOPSIS (Hwang & Yoon, 1981) untuk menggabungkan ketiga kriteria jadi satu
  CADIF_Score - jarak relatif tiap risk factor terhadap "solusi ideal" dan
  "solusi terburuk".
- ROBUSTNESS: Menambahkan fallback Epsilon untuk mencegah edge-case Division by Zero
  (menghasilkan NaN) pada dataset dengan dispersi informasi nol.

ARAH RISIKO: baik Contribution maupun Projection dibuat BERARAH (bisa negatif)
lewat perkalian/proyeksi terhadap sign(Projection_to_Target) - Projection_to_Target
sendiri adalah dot product koordinat risk factor terhadap koordinat kolom
outcome positif (lihat CA/MCA Engine). Kriteria yang berarah ini penting
supaya kategori dengan Asosiasi Negatif (proporsi outcome di bawah rata-rata) 
tidak salah diberi skor tinggi hanya karena posisinya ekstrem secara magnitude.
"""

import numpy as np
import plotly.express as px
import streamlit as st


def hitung_bobot_entropy(df_koordinat):
    """
    Menghitung bobot kriteria secara objektif menggunakan Entropy Weight Method.
    Dilengkapi pengaman (fallback) pembagian nol.
    """
    df_calc = df_koordinat.copy()

    df_calc["Directional_Contribution"] = (
        df_calc["Contribution_Gabungan_%"] * np.sign(df_calc["Projection_to_Target"])
    )

    kriteria = ["Directional_Contribution", "Cos2_Gabungan", "Projection_to_Target"]
    X = df_calc[kriteria].values.astype(float)

    m, n = X.shape

    if m <= 1:
        return {"contribution": 0.33, "cos2": 0.33, "distance": 0.34}

    X_min = X.min(axis=0)
    X_max = X.max(axis=0)

    penyebut = X_max - X_min
    penyebut[penyebut == 0] = 1e-9

    X_norm = (X - X_min) / penyebut
    X_norm = X_norm + 1e-9  

    P = X_norm / X_norm.sum(axis=0)

    k = 1.0 / np.log(m)
    E = -k * np.sum(P * np.log(P), axis=0)

    D = 1 - E

    # ==============================================================
    # FALLBACK EWM (Pencegah NaN)
    # Jika total dispersi terlalu kecil (mendekati 0), gunakan bobot merata.
    # ==============================================================
    d_sum = D.sum()
    if d_sum < 1e-9:
        W = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0])
    else:
        W = D / d_sum

    return {
        "contribution": float(W[0]),
        "cos2": float(W[1]),
        "distance": float(W[2]),
    }


def hitung_topsis(df_koordinat, bobot):
    """
    TOPSIS dengan pengaman pembagian norma dan override Asosiasi Negatif.
    """
    df_calc = df_koordinat.copy()

    df_calc["Directional_Contribution"] = (
        df_calc["Contribution_Gabungan_%"] * np.sign(df_calc["Projection_to_Target"])
    )

    kriteria = ["Directional_Contribution", "Cos2_Gabungan", "Projection_to_Target"]
    X = df_calc[kriteria].values.astype(float)

    # ==============================================================
    # PENGAMAN TOPSIS (Mencegah Norma Kolom = 0)
    # ==============================================================
    pembagi = np.sqrt((X ** 2).sum(axis=0))
    pembagi[pembagi == 0] = 1e-9  # Fallback Epsilon
    norm = X / pembagi
    
    w = np.array([bobot["contribution"], bobot["cos2"], bobot["distance"]])
    w = w / w.sum()  
    V = norm * w

    solusi_ideal_positif = V.max(axis=0)
    solusi_ideal_negatif = V.min(axis=0)

    d_plus = np.sqrt(((V - solusi_ideal_positif) ** 2).sum(axis=1))
    d_minus = np.sqrt(((V - solusi_ideal_negatif) ** 2).sum(axis=1))

    skor = d_minus / (d_plus + d_minus + 1e-9) * 10

    hasil = df_koordinat.copy()
    hasil["CADIF_Score"] = skor

    hasil["Arah_Risiko"] = np.where(
        hasil["Projection_to_Target"] > 0, "Asosiasi Positif", "Asosiasi Negatif"
    )

    return hasil.sort_values("CADIF_Score", ascending=False).reset_index(drop=True)


def run(engine_result):
    st.header("3. Risk Intelligence Engine (TOPSIS)")

    if engine_result is None:
        st.info("Menunggu hasil dari SCA/MCA Engine.")
        return None

    bobot_rekomendasi = hitung_bobot_entropy(engine_result["hasil_per_baris"])

    st.caption(
        "Metode: Kombinasi Entropy Weight Method - EWM (Shannon, 1948; Zeleny, "
        "1982) dan TOPSIS (Hwang & Yoon, 1981). Skor 0-10 dihitung dari kualitas "
        "representasi dan arah proyeksi vektor terhadap outcome."
    )

    st.subheader("Pengaturan Bobot Kriteria")
    st.info(
        "**Hybrid Mode:** Posisi awal (default) pada slider di bawah ini dihitung "
        "secara **objektif otomatis** oleh sistem menggunakan *Entropy Weight Method* "
        "(berdasarkan variansi data). Anda dapat menggesernya secara manual jika "
        "memiliki preferensi/pedoman klinis tersendiri."
    )

    c1, c2, c3 = st.columns(3)

    w_contrib = c1.slider(
        "Bobot Contribution (Pembentuk Sumbu)",
        0.0, 1.0,
        round(bobot_rekomendasi["contribution"], 2),
        0.01
    )

    w_cos2 = c2.slider(
        "Bobot Cos2 (Kualitas Visual)",
        0.0, 1.0,
        round(bobot_rekomendasi["cos2"], 2),
        0.01
    )

    w_dist = c3.slider(
        "Bobot Proyeksi (Arah Target)",
        0.0, 1.0,
        round(bobot_rekomendasi["distance"], 2),
        0.01
    )

    bobot = {"contribution": w_contrib, "cos2": w_cos2, "distance": w_dist}
    hasil_topsis = hitung_topsis(engine_result["hasil_per_baris"], bobot)

    st.write("Risk Ranking:")
    st.dataframe(
        hasil_topsis[
            [
                "risk_factor",
                "Contribution_Gabungan_%",
                "Cos2_Gabungan",
                "Projection_to_Target",
                "Arah_Risiko",
                "CADIF_Score",
            ]
        ].round(3)
    )

    fig = px.bar(
        hasil_topsis,
        x="risk_factor",
        y="CADIF_Score",
        title="Risk Priority Ranking (EWM-TOPSIS)",
        color="Arah_Risiko",
        color_discrete_map={"Asosiasi Positif": "#D85A30", "Asosiasi Negatif": "#99ccff"},
    )
    st.plotly_chart(fig, width="stretch")

    return hasil_topsis
