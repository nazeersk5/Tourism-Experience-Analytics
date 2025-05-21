import streamlit as st
import pandas as pd
import numpy as np
import os
from io import BytesIO

output_folder = "cleaned_files"
os.makedirs(output_folder, exist_ok=True)

st.title("🧼 Excel Data Cleaner App (Enhanced)")

uploaded_files = st.file_uploader("Upload Excel files", type=["xlsx"], accept_multiple_files=True)

def clean_dataset(df):
    # Normalize missing values
    df.replace(["NA", "na", "NaN", "null", "", "None"], np.nan, inplace=True)
    df.dropna(axis=1, how="all", inplace=True)  # Drop fully empty columns
    df.dropna(axis=0, how="all", inplace=True)  # Drop fully empty rows
    df_clean = df.copy()

    for col in df_clean.columns:
        if df_clean[col].dtype in [np.float64, np.int64, float, int]:
            # Fill missing
            if df_clean[col].isnull().sum() > 0:
                mean = df_clean[col].mean()
                median = df_clean[col].median()
                mode = df_clean[col].mode()[0] if not df_clean[col].mode().empty else median
                fill_value = min([(mean, 'mean'), (median, 'median'), (mode, 'mode')],
                                 key=lambda x: abs(mean - x[0]))[0]
                df_clean[col].fillna(fill_value, inplace=True)

            # IQR Outlier removal
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]

        else:
            # Categorical: fill missing with mode
            if df_clean[col].isnull().sum() > 0:
                mode = df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Unknown"
                df_clean[col].fillna(mode, inplace=True)

    return df_clean

if uploaded_files:
    for uploaded_file in uploaded_files:
        df = pd.read_excel(uploaded_file)
        st.subheader(f"Original: {uploaded_file.name}")
        st.dataframe(df.head())

        cleaned_df = clean_dataset(df)

        clean_filename = f"clean_{uploaded_file.name}"
        output_path = os.path.join(output_folder, clean_filename)
        cleaned_df.to_excel(output_path, index=False)

        st.success(f"✅ Cleaned and saved: {clean_filename}")
        st.write("Preview of Cleaned File:")
        st.dataframe(cleaned_df.head())

        # Offer as download
        towrite = BytesIO()
        cleaned_df.to_excel(towrite, index=False, engine='openpyxl')
        towrite.seek(0)
        st.download_button(label="⬇️ Download Cleaned File",
                           data=towrite,
                           file_name=clean_filename,
                           mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')