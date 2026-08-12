"""
Modul 5: Executive Summary Engine (sebelumnya Dashboard)

TITIK PERLUASAN: pecah jadi multi-page Streamlit (folder pages/) untuk
Dashboard 1-5 sesuai rancangan penuh; tambahkan export PDF (reportlab atau
weasyprint) menggantikan tombol unduh CSV di bawah.
"""

import pandas as pd
import plotly.express as px
import streamlit as st


def run(df_decision):
    st.header("5. Executive Summary")

    if df_decision is None:
        st.info("Menunggu hasil dari Decision Engine.")
        return

    col1, col2, col3 = st.columns(3)
    
    col1.metric("Jumlah Risk Factor Dianalisis", len(df_decision))
    
    top_risk = df_decision.iloc[0]
    col2.metric(
        "Top Risk", 
        top_risk["risk_factor"], 
        f"Score: {top_risk['CADIF_Score']:.2f}"
    )
    
    col3.metric("Rekomendasi Utama", top_risk["Recommendation"])

    jumlah_per_priority = df_decision["Priority"].value_counts()
    
    fig = px.pie(
        values=jumlah_per_priority.values, 
        names=jumlah_per_priority.index,
        title="Distribusi Tingkat Prioritas",
        color=jumlah_per_priority.index,
        color_discrete_map={
            "Critical": "#ff4d4d", 
            "High": "#ffa64d",
            "Medium": "#ffe066", 
            "Low": "#b3ffb3",
            "Asosiasi Negatif": "#99ccff" # <--- Diselaraskan dengan Modul 4
        },
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # GENERATOR TABEL INTERPRETASI OTOMATIS
    # ---------------------------------------------------------
    st.subheader("Tabel Interpretasi Otomatis")
    st.caption("Interpretasi di-generate secara dinamis berdasarkan parameter geometris dan multikriteria.")
    
    # Ekstraksi Top Data secara Dinamis (Menggunakan nama kolom yang baru)
    top_proyeksi = df_decision.nlargest(2, 'Projection_to_Target')['risk_factor'].tolist()
    top_cos2 = df_decision.nlargest(2, 'Cos2_Gabungan')['risk_factor'].tolist()
    top_contrib = df_decision.nlargest(1, 'Contribution_Gabungan_%')['risk_factor'].values[0]
    critical_count = len(df_decision[df_decision['Priority'] == 'Critical'])
    
    # Format teks (List to String)
    str_proyeksi = ", ".join(top_proyeksi)
    str_cos2 = ", ".join(top_cos2)
    
    # Pembuatan DataFrame Interpretasi (Bahasa Ilmiah Diperhalus)
    df_interpretasi = pd.DataFrame({
        "Aspek": [
            "Asosiasi Faktor Risiko (Proyeksi)",
            "Kualitas Representasi Visual (Cos²)",
            "Pembentuk Dimensi (Contribution)",
            "Prioritas Risiko (TOPSIS)",
            "Implikasi Kesmas"
        ],
        "Hasil Pengamatan": [
            f"Titik {str_proyeksi} berada paling searah dan terdekat dengan vektor target outcome.",
            f"{str_cos2} memiliki kualitas representasi visual tertinggi pada peta MCA.",
            f"{top_contrib} menyumbang persentase variansi terbesar pada pembentukan sumbu inersia.",
            f"Terdapat {critical_count} faktor risiko yang masuk ke dalam kategori Critical.",
            f"Berdasarkan analisis multikriteria, rekomendasi utama diarahkan pada {str_proyeksi}."
        ],
        "Interpretasi": [
            "Kategori tersebut menunjukkan kedekatan asosiasi geometris paling kuat terhadap outcome target di dalam ruang korespondensi.",
            "Titik-titik ini diproyeksikan nyaris sempurna di dimensi 2D. Posisi dan jarak visualnya sangat valid untuk diinterpretasikan.",
            "Faktor ini adalah pembeda karakteristik utama yang paling menjelaskan variasi data pada dataset yang dianalisis.",
            "Kategori ini menempati urutan teratas berdasarkan parameter model, mengindikasikan urgensi tinggi dalam kerangka sistem pengambilan keputusan ini.",
            "Kategori tersebut memperoleh prioritas penanganan relatif tertinggi secara empiris berdasarkan kriteria EWM-TOPSIS di dalam CADIF."
        ]
    })
    
    # Render Tabel di Streamlit
    st.table(df_interpretasi)
    # ---------------------------------------------------------

    st.divider()

    csv_data = df_decision.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Unduh Hasil (CSV)",
        data=csv_data,
        file_name="cadif_hasil.csv",
        mime="text/csv",
    )