"""
CADIF & MCADIF Main Application
Entry point untuk aplikasi Streamlit.

Jalankan dengan: streamlit run app.py
(atau: python -m streamlit run app.py)

Struktur project:
    app.py                          <- file ini, merangkai semua modul
    modules/
        data_management_sca.py      <- Modul 1 (SCA)
        data_management_mca.py      <- Modul 1 (MCA)
        sca_engine.py               <- Modul 2 (SCA)
        mca_engine.py               <- Modul 2 (MCA)
        risk_engine.py              <- Modul 3 (TOPSIS)
        decision_engine_sca.py      <- Modul 4 (Decision)
        decision_engine_sca.py      <- Modul 4 (Decision)
        exc_summary_sca.py          <- Modul 5 (Executive Summary)
        exc_summary_mca.py          <- Modul 5 (Executive Summary)
    config/
        decision_rules.py           <- threshold & rekomendasi (mudah diedit)
"""

import streamlit as st

# Impor Modul SCA (Bivariat)
from modules import data_management_sca
from modules import sca_engine
from modules import decision_engine_sca
from modules import exc_summary_sca

# Impor Modul MCA (Multivariat)
from modules import data_management_mca
from modules import mca_engine
from modules import decision_engine_mca
from modules import exc_summary_mca

# Impor Modul Universal (TOPSIS, Decision, Executive Summary)
from modules import risk_engine

def main():
    st.set_page_config(
        page_title="CADIF Framework",
        page_icon="⚕️",
        layout="wide",
    )

   # ==============================================================
    # INJEKSI CSS: HANYA SIDEBAR YANG BIRU NAVY & TEKS PUTIH
    # ==============================================================
    st.markdown("""
        <style>
            /* 1. Mengubah latar belakang khusus Sidebar menjadi Biru Navy ITB */
            [data-testid="stSidebar"] {
                background-color: #003057 !important;
            }
            
            /* 2. Mengubah seluruh teks di dalam Sidebar menjadi putih */
            [data-testid="stSidebar"] * {
                color: white !important;
            }
            
            /* 3. Menyesuaikan teks pada pilihan mode (radio button) */
            [data-testid="stSidebar"] .st-radio label p {
                color: white !important;
            }
        </style>
        """, unsafe_allow_html=True)
    # ==============================================================
    
    # ==============================================================
    # SIDEBAR NAVIGATION
    # ==============================================================
    st.sidebar.title("🧬 CADIF")
    st.sidebar.caption("Correspondence Analysis Decision Intelligence Framework")

    mode = st.sidebar.radio(
        "Pilih Mode Analisis:",
        [
            "📊 Bivariat AKS (1 Faktor Risiko)",
            "🕸️ Multivariat AKB (>1 Faktor Risiko)",
        ],
    )

    st.sidebar.divider()

    with st.sidebar:
        st.header("🧩 Arsitektur CADIF")
        st.caption("Alur Pemrosesan Pipa Data:")
                
        # 1. Tahap Data (Percabangan)
        st.markdown("⬇️ **Tahap 1: Data Management**")
        c1, c2 = st.columns(2)
        with c1:
            st.info("**SCA**\n\nContingency")
        with c2:
            st.info("**MCA**\n\nBurt Matrix")

        # 2. Tahap Geometri
        st.markdown("⬇️ **Tahap 2: SCA/MCA Engine**")
        st.success("**Correspondence Analysis**\n\nContrib • Cos² • Target")

        # 3. Tahap Pembobotan (Risk Intelligence)
        st.markdown("⬇️ **Tahap 3: Risk Intelligence**")
        st.warning("**Multicriteria Engine**\n\nEWM ➔ TOPSIS")

        # 4. Tahap Keputusan Klinis
        st.markdown("⬇️ **Tahap 4: Clinical Decision**")
        st.error("**Decision Engine**\n\nPriority Score & Rules")

        # 5. Output
        st.markdown("⬇️ **Tahap 5: Output**")
        st.info("**Executive Summary**\n\nDashboard & Reporting")
       
        st.markdown("---")

        # =========================================================
        # PENAMBAHAN LENCANA KONTEKS STUDI KASUS
        # =========================================================
        st.header("Lingkup Analisis Saat Ini:")
        st.info(
            "🩺 Studi Kasus Penderita Serangan Jantung"
        )
        st.markdown("---")
        # =========================================================
        
        # Identitas Pembuat
        st.caption(
            "**Dikembangkan oleh:**\n\n"
            "**Adhityo Rafael A. Sigit (10122020)**\n\n"
            "**Institut Teknologi Bandung**"
        )    

    # ==============================================================
    # RUTE 1: BIVARIATE SCA (Analisis Tunggal)
    # ==============================================================
    if mode == "📊 Bivariat AKS (1 Faktor Risiko)":
        st.title("Mode Bivariate SCA")
        st.caption(
            "Analisis univariat untuk mengevaluasi korelasi murni dari "
            "satu faktor risiko tunggal terhadap target penyakit."
        )

        df_kontingensi = data_management_sca.run()
        if df_kontingensi is not None:
            # Perubahan nama variabel dari ca_result menjadi sca_result
            sca_result = sca_engine.run(df_kontingensi)
            if sca_result is not None:
                # Masuk ke Universal Engine
                df_risk = risk_engine.run(sca_result)
                if df_risk is not None:
                    df_decision = decision_engine_sca.run(df_risk)
                    if df_decision is not None:
                        exc_summary_sca.run(df_decision)

    # ==============================================================
    # RUTE 2: MULTIVARIATE MCA (Analisis Serentak)
    # ==============================================================
    else:
        st.title("Mode Multivariate MCA")
        st.caption(
            "Analisis multivariat untuk mengurai interaksi kompleks "
            "berbagai faktor risiko secara simultan melalui Matriks Burt."
        )

        # Menggunakan modul khusus MCA
        mca_data = data_management_mca.run()
        if mca_data is not None:
            mca_result = mca_engine.run(mca_data)
            if mca_result is not None:
                # Masuk ke Universal Engine (berbagi mesin yang sama)
                df_risk = risk_engine.run(mca_result)
                if df_risk is not None:
                    df_decision = decision_engine_mca.run(df_risk)
                    if df_decision is not None:
                        exc_summary_mca.run(df_decision)


if __name__ == "__main__":
    main()
