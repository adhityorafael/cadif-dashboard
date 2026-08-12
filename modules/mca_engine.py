"""
Modul 2: MCA Engine
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def hitung_mca(input_dict):
    B = input_dict["burt_matrix"]
    N_sup = input_dict["sup_matrix"]
    target_col = input_dict["target_col"]

    # 1. Analisis pada Matriks Burt (Ruang Aktif)
    O = B.values.astype(float)
    N = O.sum()

    P = O / N
    r = P.sum(axis=1)
    c = P.sum(axis=0)

    S = (P - np.outer(r, c)) / np.sqrt(np.outer(r, c))
    U, D, Vt = np.linalg.svd(S, full_matrices=False)

    # ==============================================================
    # TRANSFORMASI SKALA BURT KE INDIKATOR
    # ==============================================================
    eigenvalues_burt = D**2          # skala Burt: lambda_Burt = lambda_indikator^2
    n_dim = min(2, len(eigenvalues_burt))
    D_indicator = np.sqrt(D[:n_dim])

    F = (U[:, :n_dim] * D_indicator) / np.sqrt(r)[:, None]
    Gamma = U[:, :n_dim] / np.sqrt(r)[:, None]
    F_burt = (U[:, :n_dim] * D[:n_dim]) / np.sqrt(r)[:, None]

    # ==============================================================
    # KOREKSI BENZECRI (Persamaan II.24)
    # Ambang batas 1/Q dievaluasi pada skala eigenvalue INDIKATOR
    # (lambda_s = sqrt(eigenvalue_Burt)), bukan langsung pada skala Burt.
    # ==============================================================
    Q = len(set(cat.split("=")[0] for cat in B.index))   # jumlah variabel aktif
    threshold = 1 / Q

    lambda_s = np.sqrt(eigenvalues_burt)   # seluruh dimensi, skala indikator

    eigenvalues_adj = np.where(
        lambda_s > threshold,
        ((Q / (Q - 1)) ** 2) * (lambda_s - threshold) ** 2,
        0.0,
    )
    n_qualifying = int(np.sum(lambda_s > threshold))
    total_inertia_adj = eigenvalues_adj.sum()

    if n_qualifying > 0:
        eigen_report = eigenvalues_adj
        total_inertia_report = total_inertia_adj
        koreksi_status = "Terkoreksi Benzecri"
    else:
        # fallback jika tidak ada dimensi yang lolos ambang, hindari div/0
        eigen_report = eigenvalues_burt
        total_inertia_report = eigenvalues_burt.sum()
        koreksi_status = "Mentah (fallback, tidak ada dimensi lolos Benzecri)"

    # ==============================================================
    # 2. Proyeksi Variabel Suplemen (Outcome)
    # ==============================================================
    N_sup_vals = N_sup.values.astype(float)
    col_totals = N_sup_vals.sum(axis=0)
    col_totals[col_totals == 0] = 1.0

    P_sup = N_sup_vals / col_totals
    G_sup = P_sup.T.dot(Gamma)
    Gamma_sup = G_sup / D_indicator

    # ==============================================================
    # 3. Ekstraksi Metrik (TIDAK terpengaruh koreksi Benzecri -
    #    dihitung langsung dari koordinat hasil SVD)
    # ==============================================================
    kontribusi = (r[:, None] * F**2) / (D_indicator**2) * 100
    jarak_total_sq = np.sum((S**2), axis=1) / r
    cos2 = (F_burt**2) / jarak_total_sq[:, None]

    target_idx = list(N_sup.columns).index(target_col)
    vektor_target = Gamma_sup[target_idx, :]

    norm_target = np.linalg.norm(vektor_target)
    if norm_target > 0:
        proyeksi = np.dot(F[:, :n_dim], vektor_target) / norm_target
    else:
        proyeksi = np.zeros(F.shape[0])

    # ==============================================================
    # 4. Susun Hasil - Cos2/CTR Gabungan memakai PENJUMLAHAN, bukan
    #    rata-rata tertimbang (sesuai konvensi cos^2 aditif, Sigma cos2=1)
    # ==============================================================
    hasil = pd.DataFrame(
        {
            "risk_factor": B.index,
            "Dim1": F[:, 0],
            "Dim2": F[:, 1] if n_dim > 1 else 0.0,
            "Contribution_Dim1_%": kontribusi[:, 0],
            "Contribution_Dim2_%": kontribusi[:, 1] if n_dim > 1 else 0.0,
            "Cos2_Dim1": cos2[:, 0],
            "Cos2_Dim2": cos2[:, 1] if n_dim > 1 else 0.0,
            "Projection_to_Target": proyeksi,
        }
    )

    hasil["Contribution_Gabungan_%"] = (
        hasil["Contribution_Dim1_%"] + hasil["Contribution_Dim2_%"]
    )
    hasil["Cos2_Gabungan"] = hasil["Cos2_Dim1"] + hasil["Cos2_Dim2"]
    hasil["Distance_to_Origin"] = np.sqrt(hasil["Dim1"] ** 2 + hasil["Dim2"] ** 2)

    # Susun Hasil Suplemen
    dist2_sup = np.sum(((P_sup.T - r) ** 2) / r, axis=1)
    cos2_sup = (G_sup**2) / dist2_sup[:, None]

    hasil_kolom = pd.DataFrame(
        {
            "Outcome_Label": N_sup.columns,
            "Dim1": Gamma_sup[:, 0],
            "Dim2": Gamma_sup[:, 1] if n_dim > 1 else 0.0,
            "Contribution_Dim1_%": 0.0,
            "Cos2_Dim1": cos2_sup[:, 0],
        }
    )

    if n_dim > 1:
        hasil_kolom["Contribution_Dim2_%"] = 0.0
        hasil_kolom["Cos2_Dim2"] = cos2_sup[:, 1]
    else:
        hasil_kolom["Contribution_Dim2_%"] = 0.0
        hasil_kolom["Cos2_Dim2"] = 0.0

    hasil_kolom["Contribution_Gabungan_%"] = 0.0
    hasil_kolom["Cos2_Gabungan"] = (
        hasil_kolom["Cos2_Dim1"] + hasil_kolom["Cos2_Dim2"]
    )

    return {
        "hasil_per_baris": hasil,
        "hasil_per_kolom": hasil_kolom,
        "eigenvalues": eigen_report,
        "pct_variance": eigen_report / total_inertia_report * 100,
        "total_inertia": total_inertia_report,
        "Q": Q,
        "threshold_benzecri": threshold,
        "n_dim_lolos_benzecri": n_qualifying,
        "koreksi_status": koreksi_status,
    }


def run(df_kontingensi):
    st.header("2. MCA Engine (True Multivariate)")

    if df_kontingensi is None:
        st.info("Menunggu data Matriks Burt.")
        return None

    mca_result = hitung_mca(df_kontingensi)

    c1, c2 = st.columns(2)
    c1.metric("Total Inertia (Active)", f"{mca_result['total_inertia']:.4f}")

    variance_sum = mca_result["pct_variance"][:2].sum()
    c2.metric("Variansi Dim1+Dim2", f"{variance_sum:.1f}%")

    st.caption(
        f"Metode: {mca_result['koreksi_status']} | "
        f"Ambang 1/Q = {mca_result['threshold_benzecri']:.4f} | "
        f"Dimensi lolos: {mca_result['n_dim_lolos_benzecri']} dari {len(mca_result['eigenvalues'])}"
    )

    df_plot_baris = mca_result["hasil_per_baris"]

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
        title="Peta MCA Berbasis Matriks Burt (Suplemen Target)",
    )

    fig.update_traces(
        marker=dict(size=12, color="#1f77b4"),
        textposition="bottom center",
        name="Faktor Risiko (Active)",
        showlegend=True,
    )

    df_plot_kolom = mca_result["hasil_per_kolom"]
    fig.add_scatter(
        x=df_plot_kolom["Dim1"],
        y=df_plot_kolom["Dim2"],
        mode="markers+text",
        text=df_plot_kolom["Outcome_Label"],
        textposition="top center",
        marker=dict(size=20, color="#d62728", symbol="star"),
        name="Outcome (Supplementary)",
        customdata=df_plot_kolom[["Cos2_Gabungan"]],
        hovertemplate=(
            "<b>Target: %{text}</b><br>"
            "Cos2: %{customdata[0]:.3f}<br>"
            "<i>(Suplemen: 0% Contrib)</i><extra></extra>"
        ),
    )

    fig.add_hline(y=0, line_color="gray", line_width=1, line_dash="dot")
    fig.add_vline(x=0, line_color="gray", line_width=1, line_dash="dot")

    fig.update_layout(height=800)
    st.plotly_chart(fig, use_container_width=True)

    return mca_result