"""
Modul 1: Data Management (MCADIF - Supplementary Outcome)
Membangun Matriks Burt dari variabel aktif dan memisahkan Outcome 
sebagai Suplemen.
"""

import numpy as np
import pandas as pd
import streamlit as st


def binorisasi_variabel_numerik(df):
    df_processed = df.copy()
    kolom_numerik = [
        "BMI",
        "HeightInMeters",
        "WeightInKilograms",
        "PhysicalHealthDays",
        "MentalHealthDays",
        "SleepHours",
    ]

    for col in kolom_numerik:
        if col in df_processed.columns:
            if df_processed[col].dtype == "object":
                df_processed[col] = (
                    df_processed[col].astype(str).str.replace(",", ".")
                )
            df_processed[col] = pd.to_numeric(
                df_processed[col], errors="coerce"
            )

    # 1. BMI (Standar WHO)
    if "BMI" in df_processed.columns:
        bins_bmi = [-np.inf, 18.5, 25.0, 30.0, np.inf]
        labels_bmi = ["Underweight", "Normal", "Overweight", "Obese"]
        df_processed["BMI_Cat"] = pd.cut(
            df_processed["BMI"], bins=bins_bmi, labels=labels_bmi
        )
        # Hapus kolom numerik aslinya
        df_processed.drop(columns=["BMI"], inplace=True)

    # 2. SleepHours (Standar National Sleep Foundation / CDC)
    if "SleepHours" in df_processed.columns:
        bins_sleep = [-np.inf, 6.9, 9.0, np.inf]
        labels_sleep = [
            "Short_Sleep (<7h)",
            "Recommended (7-9h)",
            "Long_Sleep (>9h)",
        ]
        df_processed["SleepHours_Cat"] = pd.cut(
            df_processed["SleepHours"], bins=bins_sleep, labels=labels_sleep
        )
        df_processed.drop(columns=["SleepHours"], inplace=True)

    # 3. PhysicalHealthDays (Standar CDC BRFSS / 14-day rule)
    if "PhysicalHealthDays" in df_processed.columns:
        bins_phys = [-np.inf, 0.5, 13.5, np.inf]
        labels_phys = [
            "None (0 days)",
            "Occasional (1-13 days)",
            "Frequent Distress (14-30 days)",
        ]
        df_processed["PhysicalHealth_Cat"] = pd.cut(
            df_processed["PhysicalHealthDays"], bins=bins_phys, labels=labels_phys
        )
        df_processed.drop(columns=["PhysicalHealthDays"], inplace=True)

    # 4. MentalHealthDays (Standar CDC BRFSS / 14-day rule)
    if "MentalHealthDays" in df_processed.columns:
        bins_ment = [-np.inf, 0.5, 13.5, np.inf]
        labels_ment = [
            "None (0 days)",
            "Occasional (1-13 days)",
            "Frequent Distress (14-30 days)",
        ]
        df_processed["MentalHealth_Cat"] = pd.cut(
            df_processed["MentalHealthDays"], bins=bins_ment, labels=labels_ment
        )
        df_processed.drop(columns=["MentalHealthDays"], inplace=True)

    # 5. HeightInMeters (Antropometri Populasi Dewasa)
    if "HeightInMeters" in df_processed.columns:
        bins_height = [-np.inf, 1.60, 1.75, np.inf]
        labels_height = ["Short (<1.60m)", "Average (1.60-1.75m)", "Tall (>1.75m)"]
        df_processed["Height_Cat"] = pd.cut(
            df_processed["HeightInMeters"], bins=bins_height, labels=labels_height
        )
        df_processed.drop(columns=["HeightInMeters"], inplace=True)

    # 6. WeightInKilograms (Deskriptif Populasi)
    if "WeightInKilograms" in df_processed.columns:
        bins_weight = [-np.inf, 55.0, 80.0, np.inf]
        labels_weight = ["Light (<55kg)", "Normal (55-80kg)", "Heavy (>80kg)"]
        df_processed["Weight_Cat"] = pd.cut(
            df_processed["WeightInKilograms"], bins=bins_weight, labels=labels_weight
        )
        df_processed.drop(columns=["WeightInKilograms"], inplace=True)

    return df_processed


def run():
    st.header("1. Data Management (MCADIF)")
    st.info(
        "Mode Burt Matrix: Faktor risiko akan disilangkan satu sama lain "
        "(Multivariat). Outcome akan diproyeksikan sebagai Variabel Suplemen."
    )

    df_kontingensi = None
    file = st.file_uploader(
        "Upload data mentah (CSV)", type=["csv"], key="upload_mca"
    )

    if file is not None:
        df_raw = pd.read_csv(file)
        df_clean = binorisasi_variabel_numerik(df_raw)

        # 1. Pilih Outcome (Supplementary)
        kolom_outcome = st.selectbox(
            "Pilih kolom TARGET (Outcome / Suplemen)", df_clean.columns
        )
        nilai_outcome = df_clean[kolom_outcome].dropna().unique()
        target_value = st.selectbox(
            "Pilih nilai target 'BAHAYA / POSITIF'", nilai_outcome
        )

        # 2. Pilih Faktor Risiko (Active) - Pilih minimal 2 untuk membentuk MCA!
        kolom_risk_factors = st.multiselect(
            "Pilih kolom risk factor (Pilih >1 Variabel untuk melihat "
            "penyebaran MCA)",
            [c for c in df_clean.columns if c != kolom_outcome],
        )

        if kolom_outcome and target_value and len(kolom_risk_factors) > 1:
            df_subset = df_clean[kolom_risk_factors + [kolom_outcome]].dropna()

            # =========================================================
            # PEMBUATAN MATRIKS BURT (DIJAMIN NUMERIK & NAMA COCOK)
            # =========================================================
            Z_active = pd.get_dummies(
                df_subset[kolom_risk_factors], prefix_sep="="
            ).astype(int)

            Z_sup = pd.get_dummies(
                df_subset[[kolom_outcome]], prefix_sep="="
            ).astype(int)

            # Lakukan Dot Product
            B = Z_active.T.dot(Z_active)
            N_sup = Z_active.T.dot(Z_sup)
            # =========================================================

            st.success(
                "Matriks Burt berhasil dibuat! Ruang multivariat siap dianalisis."
            )
            col1, col2 = st.columns(2)
            
            col1.write("Burt Matrix (Sebagian):")
            col1.dataframe(B.iloc[:5, :5])
            
            col2.write("Sup Matrix:")
            col2.dataframe(N_sup.iloc[:5, :])

            df_kontingensi = {
                "burt_matrix": B,
                "sup_matrix": N_sup,
                "target_col": f"{kolom_outcome}={target_value}",
            }

    return df_kontingensi