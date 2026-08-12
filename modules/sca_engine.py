"""
Modul 2: Simple Correspondence Analysis (SCA) Engine

Kontrak: menerima tabel kontingensi (DataFrame numerik), mengembalikan dict
berisi principal coordinates, contribution, cos2, distance, inertia per baris.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def hitung_sca(tabel):
    O = tabel.values.astype(float)
    N = O.sum()

    r = O.sum(axis=1) / N
    c = O.sum(axis=0) / N

    P = O / N
    S = (P - np.outer(r, c)) / np.sqrt(np.outer(r, c))

    U, D, Vt = np.linalg.svd(S, full_matrices=False)
    eigenvalues = D**2
    total_inertia = eigenvalues.sum()

    n_dim = min(2, len(eigenvalues))

    # =================================================================
    # IMPLEMENTASI ASYMMETRIC BIPLOT (ROW METRIC PRESERVING)
    # =================================================================
    # 1. BARIS: Koordinat Utama (Principal) untuk Faktor Risiko
    F = (U[:, :n_dim] * D[:n_dim]) / np.sqrt(r)[:, None]

    # 2. KOLOM: Koordinat Standar (Standard) untuk Vektor Outcome & Plotting
    Gamma = (Vt.T[:, :n_dim]) / np.sqrt(c)[:, None]

    # 3. KOLOM: Koordinat Utama (Principal) - Hanya untuk Metrik (Contrib/Cos2)
    G = (Vt.T[:, :n_dim] * D[:n_dim]) / np.sqrt(c)[:, None]
    # =================================================================

    max_dim = min(tabel.shape[0], tabel.shape[1]) - 1
    if max_dim == 1 and F.shape[1] > 1:
        F[:, 1] = 0.0
        Gamma[:, 1] = 0.0
        G[:, 1] = 0.0

    # Metrik Baris
    kontribusi = (r[:, None] * F**2) / eigenvalues[:n_dim] * 100
    jarak_total_sq = np.sum((S**2), axis=1) / r
    cos2 = (F**2) / jarak_total_sq[:, None]

    # PROYEKSI DOT PRODUCT (UTAMA x STANDAR)
    vektor_target = Gamma[-1, :]
    norm_target = np.linalg.norm(vektor_target)
    if norm_target > 0:
        proyeksi = np.dot(F[:, :n_dim], vektor_target) / norm_target
    else:
        proyeksi = np.zeros(F.shape[0])

    hasil = pd.DataFrame(
        {
            "risk_factor": tabel.index,
            "Dim1": F[:, 0],
            "Dim2": F[:, 1] if n_dim > 1 else 0.0,
            "Contribution_Dim1_%": kontribusi[:, 0],
            "Contribution_Dim2_%": kontribusi[:, 1] if n_dim > 1 else 0.0,
            "Cos2_Dim1": cos2[:, 0],
            "Cos2_Dim2": cos2[:, 1] if n_dim > 1 else 0.0,
        }
    )

    bobot1 = eigenvalues[0] / eigenvalues[:n_dim].sum()
    bobot2 = eigenvalues[1] / eigenvalues[:n_dim].sum() if n_dim > 1 else 0.0

    hasil["Contribution_Gabungan_%"] = (
        hasil["Contribution_Dim1_%"] * bobot1
        + hasil["Contribution_Dim2_%"] * bobot2
    )
    hasil["Cos2_Gabungan"] = (
        hasil["Cos2_Dim1"] * bobot1 + hasil["Cos2_Dim2"] * bobot2
    )
    hasil["Distance_to_Origin"] = np.sqrt(hasil["Dim1"] ** 2 + hasil["Dim2"] ** 2)
    hasil["Projection_to_Target"] = proyeksi

    # Metrik Kolom (Menggunakan G untuk Contrib/Cos2, Gamma untuk Visual Dim1/Dim2)
    kontribusi_kolom = (c[:, None] * G**2) / eigenvalues[:n_dim] * 100
    jarak_total_sq_kolom = np.sum((S**2), axis=0) / c
    cos2_kolom = (G**2) / jarak_total_sq_kolom[:, None]

    hasil_kolom = pd.DataFrame(
        {
            "Outcome_Label": tabel.columns,
            "Dim1": Gamma[:, 0],  # <-- Koordinat Standar untuk Visual
            "Dim2": Gamma[:, 1] if n_dim > 1 else 0.0,
            "Contribution_Dim1_%": kontribusi_kolom[:, 0],
            "Cos2_Dim1": cos2_kolom[:, 0],
        }
    )

    if n_dim > 1:
        hasil_kolom["Contribution_Dim2_%"] = kontribusi_kolom[:, 1]
        hasil_kolom["Cos2_Dim2"] = cos2_kolom[:, 1]
    else:
        hasil_kolom["Contribution_Dim2_%"] = 0.0
        hasil_kolom["Cos2_Dim2"] = 0.0

    hasil_kolom["Contribution_Gabungan_%"] = (
        hasil_kolom["Contribution_Dim1_%"] * bobot1
        + hasil_kolom["Contribution_Dim2_%"] * bobot2
    )
    hasil_kolom["Cos2_Gabungan"] = (
        hasil_kolom["Cos2_Dim1"] * bobot1 + hasil_kolom["Cos2_Dim2"] * bobot2
    )

    return {
        "hasil_per_baris": hasil,
        "hasil_per_kolom": hasil_kolom,
        "eigenvalues": eigenvalues,
        "pct_variance": eigenvalues / total_inertia * 100,
        "total_inertia": total_inertia,
    }


def run(df_kontingensi):
    st.header("2. SCA Engine (Bivariat)")

    if df_kontingensi is None or df_kontingensi.shape[0] < 2:
        st.info("Menunggu data kontingensi yang valid dari Modul 1.")
        return None

    sca_result = hitung_sca(df_kontingensi)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Inertia", f"{sca_result['total_inertia']:.4f}")
    with col2:
        variance_sum = sca_result["pct_variance"][:2].sum()
        st.metric("Variansi Dim1+Dim2", f"{variance_sum:.1f}%")

    st.write("Hasil SCA per risk factor:")
    st.dataframe(sca_result["hasil_per_baris"].round(3))

    # -------------------------------------------------------------
    # PEMBUATAN BIPLOT (BARIS & KOLOM)
    # -------------------------------------------------------------
    df_plot_baris = sca_result["hasil_per_baris"]

    # Layer 1: Plot Faktor Risiko (Titik Biru)
    fig = px.scatter(
        df_plot_baris,
        x="Dim1",
        y="Dim2",
        text="risk_factor",
        hover_data={
            "Contribution_Gabungan_%": ":.2f",
            "Cos2_Gabungan": ":.3f",
            "Projection_to_Target": ":.3f",
            "Dim1": False,
            "Dim2": False,
        },
        title="Asymmetric Biplot SCA (Row-Principal) - Arah Proyeksi Risiko",
    )
    
    # Percantik titik faktor risiko
    fig.update_traces(
        marker=dict(size=12, color="#1f77b4"),
        textposition="bottom center",
        name="Faktor Risiko",
        showlegend=True,
    )

    # Layer 2: Plot Outcome Target / Penyakit (Bintang Merah)
    df_plot_kolom = sca_result["hasil_per_kolom"]
    fig.add_scatter(
        x=df_plot_kolom["Dim1"],
        y=df_plot_kolom["Dim2"],
        mode="markers+text",
        text=df_plot_kolom["Outcome_Label"],
        textposition="top center",
        marker=dict(size=18, color="#d62728", symbol="star"),
        name="Outcome (Penyakit)",
        # --- TAMBAHAN TOOLTIP METRIK ---
        customdata=df_plot_kolom[["Contribution_Gabungan_%", "Cos2_Gabungan"]],
        hovertemplate=(
            "<b>Titik Outcome: %{text}</b><br>"
            "Contrib_Gabungan_%: %{customdata[0]:.2f}<br>"
            "Cos2_Gabungan: %{customdata[1]:.3f}<extra></extra>"
        ),
        # -------------------------------
    )

    # Tambahkan garis Kuadran (0,0)
    fig.add_hline(y=0, line_color="gray", line_width=1, line_dash="dot")
    fig.add_vline(x=0, line_color="gray", line_width=1, line_dash="dot")

    # Tampilkan di Streamlit
    st.plotly_chart(fig, use_container_width=True)

    return sca_result