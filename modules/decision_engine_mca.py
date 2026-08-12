"""
Modul 4: Decision Engine

Kontrak: Menerima DataFrame dengan kolom CADIF_Score, mengembalikan DataFrame
ditambah kolom Priority dan Recommendation.

METODE KLASIFIKASI: Kuartil (Q1/Q2/Q3) dari seluruh distribusi CADIF_Score.
Batas kategori dihitung otomatis dari data yang sedang dianalisis, sehingga 
selalu relevan terhadap sebaran risk factor pada dataset tersebut.

Referensi metode: pendekatan kuartil untuk klasifikasi data epidemiologi
terbukti kompetitif/lebih akurat dibanding metode lain pada choropleth data
kesehatan (Brewer & Pickle, 2002, Annals of the American Association of
Geographers). 

Pembagian:
    CADIF_Score <= Q1               -> Low
    Q1 < CADIF_Score <= Q2 (median) -> Medium
    Q2 < CADIF_Score <= Q3          -> High
    CADIF_Score > Q3                -> Critical
"""

import pandas as pd
import streamlit as st

REKOMENDASI_PER_LABEL = {
    "Critical": "Immediate Intervention",
    "High": "Targeted Prevention Program",
    "Medium": "Continuous Monitoring",
    "Low": "Routine Observation",
}


def hitung_breakpoint_kuartil(skor: pd.Series):
    """Mengembalikan (Q1, Q2/median, Q3) dari suatu Series CADIF_Score."""
    q1 = skor.quantile(0.25)
    q2 = skor.quantile(0.50)
    q3 = skor.quantile(0.75)
    return q1, q2, q3


def terapkan_decision_rules(df_risk):
    if df_risk is None or len(df_risk) == 0:
        return df_risk

    hasil = df_risk.copy()
    
    # ---- LOGIKA DISTRIBUSI KUARTIL KESELURUHAN ----
    # Kuartil dihitung dari seluruh populasi Risk Score untuk menjaga
    # integritas distribusi statistik secara utuh.
    q1, q2, q3 = hitung_breakpoint_kuartil(hasil["CADIF_Score"])

    def klasifikasi(skor):
        if skor <= q1:
            label = "Low"
        elif skor <= q2:
            label = "Medium"
        elif skor <= q3:
            label = "High"
        else:
            label = "Critical"
        return label, REKOMENDASI_PER_LABEL[label]

    hasil["Priority"], hasil["Recommendation"] = zip(
        *hasil["CADIF_Score"].apply(klasifikasi)
    )

    # ---- OVERRIDE UNTUK KATEGORI ASOSIASI NEGATIF ----
    # Meskipun nilai Risk Score-nya tinggi secara matematis (dan mungkin 
    # melewati batas Q2 atau Q3), jika arahnya negatif (menjauhi target), 
    # sistem akan menimpa paksa (override) prioritasnya agar tidak salah 
    # dianggap sebagai faktor bahaya.
    if "Arah_Risiko" in hasil.columns:
        mask_negatif = hasil["Arah_Risiko"] == "Asosiasi Negatif"
        hasil.loc[mask_negatif, "Priority"] = "Asosiasi Negatif"
        hasil.loc[mask_negatif, "Recommendation"] = (
            "Health Promotion or Awareness"
        )

    hasil.attrs["breakpoint_kuartil"] = (q1, q2, q3)
    return hasil


def run(df_risk):
    st.header("4. Decision Engine")

    if df_risk is None:
        st.info("Menunggu hasil dari Risk Intelligence Engine.")
        return None

    st.info(
        "Metode klasifikasi: KUARTIL dari distribusi keseluruhan Risk_Score (Q1/Q2/Q3). "
        "Faktor yang memiliki 'Asosiasi Negatif' akan di-override secara klinis. "
        "**Kategori prioritas bersifat relatif terhadap distribusi CADIF Score pada data yang dianalisis, bukan ambang klinis.**"
    )

    df_decision = terapkan_decision_rules(df_risk)
    q1, q2, q3 = df_decision.attrs.get("breakpoint_kuartil", (None, None, None))
    if q1 is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Q1 (batas Low/Medium)", f"{q1:.2f}")
        c2.metric("Q2 / Median (batas Medium/High)", f"{q2:.2f}")
        c3.metric("Q3 (batas High/Critical)", f"{q3:.2f}")

    def warna_priority(val):
        peta = {
            "Critical": "background-color: #ff4d4d",
            "High": "background-color: #ffa64d",
            "Medium": "background-color: #ffe066",
            "Low": "background-color: #b3ffb3",
            "Asosiasi Negatif": "background-color: #99ccff",
        }
        return peta.get(val, "")

    st.write("Decision Matrix:")

    st.dataframe(
        df_decision[
            ["risk_factor", "CADIF_Score", "Priority", "Recommendation"]
        ]
        .style.map(warna_priority, subset=["Priority"])
        .format({"CADIF_Score": "{:.2f}"})
    )

    return df_decision