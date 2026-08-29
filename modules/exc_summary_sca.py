"""
Modul 5: Executive Summary Engine (Khusus SCA / Bivariat)
Menggunakan visualisasi Pie Chart & Tabel Interpretasi Otomatis terbaru.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

def run(df_decision):
    st.header("5. Executive Summary")

    if df_decision is None:
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah Kategori Dianalisis", len(df_decision))
    
    top_risk = df_decision.iloc[0]
    col2.metric("Top Risk", top_risk["risk_factor"], f"Score: {top_risk['CADIF_Score']:.2f}")
    col3.metric("Rekomendasi Utama", top_risk["Recommendation"])

    # --- PIE CHART ---
    jumlah_per_priority = df_decision["Priority"].value_counts()
    fig = px.pie(
        values=jumlah_per_priority.values, 
        names=jumlah_per_priority.index,
        title="Distribusi Tingkat Prioritas",
        color=jumlah_per_priority.index,
        color_discrete_map={
            "Critical": "#ff4d4d", "High": "#ffa64d", "Medium": "#ffe066", 
            "Low": "#b3ffb3", "Asosiasi Negatif": "#99ccff"
        },
    )
    st.plotly_chart(fig, use_container_width=True)

    csv_data = df_decision.to_csv(index=False).encode("utf-8")
    st.download_button(label="Unduh Hasil (CSV)", data=csv_data, file_name="sca_hasil.csv", mime="text/csv")
