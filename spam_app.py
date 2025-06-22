import streamlit as st
import joblib
import pandas as pd
import os

# Load model and selected features
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model_svm.pkl')
FEATURE_PATH = os.path.join(os.path.dirname(__file__), 'selected_features.pkl')

model = joblib.load(MODEL_PATH)
selected_features = joblib.load(FEATURE_PATH)

# Streamlit UI
st.title("📧 Email Spam Classifier")
st.markdown("Ketik isi email di bawah ini untuk mengetahui apakah tergolong **SPAM** atau **HAM** (bukan spam).")

sentence = st.text_input("Masukkan kalimat email:")

if sentence.strip():
    with st.spinner("Menganalisis..."):
        # Preprocessing
        words = sentence.lower().split()

        # Manual BoW based on selected features
        input_dict = {word: 0 for word in selected_features}
        for word in words:
            if word in input_dict:
                input_dict[word] += 1

        input_df = pd.DataFrame([input_dict])

        # Predict
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        if pred == 1:
            st.error(f"🚫 Prediksi: SPAM (Probabilitas: {prob:.4f})")
        else:
            st.success(f"✅ Prediksi: HAM (Probabilitas spam: {prob:.4f})")
else:
    st.info("Masukkan teks email terlebih dahulu.")
