"""
Modul 1: Data Management (SCA - Bivariate)

Kontrak: Menerima file CSV (via Streamlit uploader), mengembalikan tabel
kontingensi bivariat murni (1 Risk Factor x 1 Outcome), 
siap dipakai modul SCA Engine.
"""

import numpy as np
import pandas as pd
import streamlit as st


def binorisasi_variabel_numerik(df):
    """
    Mengubah 6 variabel numerik menjadi kategorikal berdasarkan standar 
    klinis/epidemiologis sebelum masuk ke mesin Contingency Table SCA.
    """
    df_processed = df.copy()

    # =================================================================
    # PEMBERSIHAN FORMAT ANGKA (MENGATASI KOMA DESIMAL)
    # =================================================================
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

    # =================================================================
    # PROSES BINORISASI
    # =================================================================
    # 1. BMI (Standar WHO)
    if "BMI" in df_processed.columns:
        bins_bmi = [-np.inf, 18.5, 25.0, 30.0, np.inf]
        labels_bmi = ["Underweight", "Normal", "Overweight", "Obese"]
        df_processed["BMI_Cat"] = pd.cut(
            df_processed["BMI"], bins=bins_bmi, labels=labels_bmi
        )
        df_processed.drop(columns=["BMI"], inplace=True)

    # 2. SleepHours (Standar National Sleep Foundation / CDC)
    if "SleepHours" in df_processed.columns:
        bins_sleep = [-np.inf, 6.9, 9.0, np.inf]
        labels_sleep = [
            "Short_Sleep (<7h)", 
            "Recommended (7-9h)", 
            "Long_Sleep (>9h)"
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
    st.header("1. Data Management (Bivariat SCA)")

    mode_input = st.radio(
        "Sumber data",
        [
            "Tabel kontingensi langsung (CSV)",
            "Data mentah (hitung kontingensi otomatis)",
        ],
        help=(
            "Tabel kontingensi: baris = kategori risk factor, "
            "kolom = kategori outcome (mis. Heart Attack/No Heart Attack)."
        ),
    )

    df_kontingensi = None

    if mode_input == "Tabel kontingensi langsung (CSV)":
        file = st.file_uploader(
            "Upload tabel kontingensi (CSV)", type=["csv"], key="upload_direct"
        )
        if file is not None:
            df_raw = pd.read_csv(file, index_col=0)
            st.write("Pratinjau data:")
            st.dataframe(df_raw)

            na_count = df_raw.isna().sum().sum()
            if na_count > 0:
                st.warning(
                    f"Ditemukan {na_count} sel kosong. "
                    "Baris dengan NA akan dibuang."
                )
                df_raw = df_raw.dropna()

            if (df_raw.select_dtypes(include=[np.number]) < 0).any().any():
                st.error(
                    "Ditemukan nilai negatif pada tabel kontingensi - "
                    "data tidak valid."
                )
                return None
            
            # FITUR BARU: Pilih Target Bahaya pada Mode Upload Kontingensi
            target_value = st.selectbox(
                "Pilih kolom target 'BAHAYA/POSITIF'", df_raw.columns
            )
            
            if target_value:
                # Susun ulang agar target berada di kolom paling kanan
                cols = list(df_raw.columns)
                if target_value in cols:
                    cols.remove(target_value)
                    cols.append(target_value)
                    df_kontingensi = df_raw[cols]

    else:
        file = st.file_uploader(
            "Upload data mentah (CSV)", type=["csv"], key="upload_raw"
        )
        if file is not None:
            df_raw = pd.read_csv(file)
            st.write(f"Ukuran data: {df_raw.shape[0]} baris, {df_raw.shape[1]} kolom")

            # ========================================================
            # EKSEKUSI FUNGSI BINORISASI
            # ========================================================
            df_clean = binorisasi_variabel_numerik(df_raw)

            kolom_outcome = st.selectbox(
                "Pilih kolom TARGET (Outcome)", df_clean.columns
            )
            
            # Penentuan Target untuk Proyeksi Vektor
            unique_targets = df_clean[kolom_outcome].dropna().unique()
            target_value = st.selectbox(
                "Pilih nilai target 'BAHAYA/POSITIF'", unique_targets
            )

            # PERBAIKAN BIVARIAT MURNI: Mengganti multiselect menjadi selectbox tunggal
            kolom_risk_factor = st.selectbox(
                "Pilih 1 kolom Risk Factor (Bivariat Murni)",
                [c for c in df_clean.columns if c != kolom_outcome],
            )

            if kolom_outcome and target_value and kolom_risk_factor:
                # Pembuatan tabel kontingensi 1v1
                df_kontingensi = pd.crosstab(df_clean[kolom_risk_factor], df_clean[kolom_outcome])
                
                # Memberikan nama variabel sebagai awalan agar rapi di grafik
                df_kontingensi.index = [f"{kolom_risk_factor}={idx}" for idx in df_kontingensi.index]
                
                # REORDER COLUMNS: Pastikan target_value berada di kolom paling akhir (kanan)
                cols = list(df_kontingensi.columns)
                if target_value in cols:
                    cols.remove(target_value)
                    cols.append(target_value)
                    df_kontingensi = df_kontingensi[cols]

                st.write(f"Tabel kontingensi murni ({kolom_risk_factor} vs {kolom_outcome}):")
                st.dataframe(df_kontingensi)

    return df_kontingensi