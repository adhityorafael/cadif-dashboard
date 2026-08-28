"""
CADIF & MCADIF Main Application
Entry point untuk aplikasi Streamlit.
"""

import streamlit as st

# Impor Langsung (Tanpa 'from modules') karena file menumpuk di halaman depan GitHub
import data_management_sca
import sca_engine
import decision_engine_sca
import exc_summary_sca

import data_management_mca
import mca_engine
import decision_engine_mca
import exc_summary_mca

import risk_engine

def main():
    st.set_page_config(
        page_title="CADIF Framework",
        page_icon="⚕️",
        layout="wide",
    )

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
                
        st.markdown("⬇️ **Tahap 1: Data Management**")
        c1, c2 = st.columns(2)
        with c1:
            st.info("**SCA**\n\nContingency")
        with c2:
            st.info("**MCA**\n\nBurt Matrix")

        st.markdown("⬇️ **Tahap 2: SCA/MCA Engine**")
        st.success("**Correspondence Analysis**\n\nContrib • Cos² • Target")

        st.markdown("⬇️ **Tahap 3: Risk Intelligence**")
        st.warning("**Multicriteria Engine**\n\nEWM ➔ TOPSIS")

        st.markdown("⬇️ **Tahap 4: Clinical Decision**")
        st.error("**Decision Engine**\n\nPriority Score & Rules")

        st.markdown("⬇️ **Tahap 5: Output**")
        st.info("**Executive Summary**\n\nDashboard & Reporting")
       
        st.markdown("---")
        st.header("Lingkup Analisis Saat Ini:")
        st.info("🩺 Studi Kasus Penderita Serangan Jantung")
        st.markdown("---")
        
        st.caption(
            "**Dikembangkan oleh:**\n\n"
            "**Adhityo Rafael A. Sigit (10122020)**\n\n"
            "**Institut Teknologi Bandung**"
        )    

    # ==============================================================
    # RUTE 1: BIVARIATE SCA
    # ==============================================================
    if mode == "📊 Bivariat AKS (1 Faktor Risiko)":
        st.title("Mode Bivariate SCA")
        st.caption("Analisis univariat untuk mengevaluasi korelasi murni dari satu faktor risiko tunggal terhadap target penyakit.")

        df_kontingensi = data_management_sca.run()
        if df_kontingensi is not None:
            sca_result = sca_engine.run(df_kontingensi)
            if sca_result is not None:
                df_risk = risk_engine.run(sca_result)
                if df_risk is not None:
                    df_decision = decision_engine_sca.run(df_risk)
                    if df_decision is not None:
                        exc_summary_sca.run(df_decision)

    # ==============================================================
    # RUTE 2: MULTIVARIATE MCA DENGAN TABS
    # ==============================================================
    else:
        st.title("Mode Multivariate MCA")
        st.caption("Analisis multivariat untuk mengurai interaksi kompleks berbagai faktor risiko secara simultan melalui Matriks Burt.")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📂 1. Data", 
            "🧭 2. MCA Engine", 
            "⚖️ 3. TOPSIS", 
            "🚦 4. Prioritas", 
            "📈 5. Summary"
        ])

        with tab1:
            mca_data = data_management_mca.run()
        
        with tab2:
            if mca_data is not None:
                mca_result = mca_engine.run(mca_data)
            else:
                st.info("Selesaikan unggah data di Tab 1 terlebih dahulu.")
                mca_result = None
                
        with tab3:
            if mca_result is not None:
                df_risk = risk_engine.run(mca_result)
            else:
                st.info("Selesaikan Analisis MCA di Tab 2 terlebih dahulu.")
                df_risk = None
                
        with tab4:
            if df_risk is not None:
                df_decision = decision_engine_mca.run(df_risk)
            else:
                st.info("Selesaikan perhitungan TOPSIS di Tab 3 terlebih dahulu.")
                df_decision = None
                
        with tab5:
            if df_decision is not None:
                exc_summary_mca.run(df_decision)

if __name__ == "__main__":
    main()
