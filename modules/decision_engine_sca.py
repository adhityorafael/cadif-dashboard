"""
Modul 4: Decision Engine (Khusus SCA / Bivariat)
Klasifikasi prioritas tetap berdasarkan CADIF Score.
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
    q1 = skor.quantile(0.25)
    q2 = skor.quantile(0.50)
    q3 = skor.quantile(0.75)
    return q1, q2, q3

def terapkan_decision_rules(df_risk):
    if df_risk is None or len(df_risk) == 0:
        return df_risk

    hasil = df_risk.copy()
    q1, q2, q3 = hitung_breakpoint_kuartil(hasil["CADIF_Score"])

    def klasifikasi(skor):
        if skor <= q1: return "Low"
        elif skor <= q2: return "Medium"
        elif skor <= q3: return "High"
        else: return "Critical"

    hasil["Priority"] = hasil["CADIF_Score"].apply(klasifikasi)
    hasil["Recommendation"] = hasil["Priority"].map(REKOMENDASI_PER_LABEL)

    # Override Asosiasi Negatif
    if "Arah_Risiko" in hasil.columns:
        mask_negatif = hasil["Arah_Risiko"] == "Asosiasi Negatif"
        hasil.loc[mask_negatif, "Priority"] = "Asosiasi Negatif"
        hasil.loc[mask_negatif, "Recommendation"] = "Health Promotion or Awareness"

    hasil.attrs["breakpoint_kuartil"] = (q1, q2, q3)
    return hasil

def run(df_risk):
    st.header("4. Decision Engine")

    if df_risk is None:
        return None

    df_decision = terapkan_decision_rules(df_risk)
    q1, q2, q3 = df_decision.attrs.get("breakpoint_kuartil", (0, 0, 0))
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Q1 (Batas Low/Medium)", f"{q1:.2f}")
    c2.metric("Q2 / Median (Batas Medium/High)", f"{q2:.2f}")
    c3.metric("Q3 (Batas High/Critical)", f"{q3:.2f}")

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
        df_decision[["risk_factor", "CADIF_Score", "Priority", "Recommendation"]]
        .style.map(warna_priority, subset=["Priority"])
        .format({"CADIF_Score": "{:.2f}"})
    )

    return df_decision