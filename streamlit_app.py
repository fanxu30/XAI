import streamlit as st
import pandas as pd
import shap
import joblib
from pathlib import Path

st.set_page_config(page_title="Resume Categorizer", layout="wide")
st.title("Resume Categorization with Explanations")

model = joblib.load(Path("/Users/fanxu/XAI/resume_model.joblib"))
# Use simple text masker to avoid tokenizer expectations and explain predict_proba
text_masker = shap.maskers.Text()
explainer = shap.Explainer(model.predict_proba, masker=text_masker, output_names=model.classes_)

st.sidebar.header("Input")
resume_text = st.sidebar.text_area("Paste resume text", height=300)

if st.sidebar.button("Predict"):
    if not resume_text.strip():
        st.warning("Please paste a resume.")
    else:
        pred = model.predict([resume_text])[0]
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([resume_text])[0]
        st.subheader(f"Predicted category: {pred}")
        if proba is not None:
            prob_df = pd.DataFrame([proba], columns=model.classes_).T
            prob_df.columns = ["probability"]
            st.bar_chart(prob_df)

        st.markdown("### Explanation")
        try:
            shap_values = explainer([resume_text])
            target_class_idx = list(model.classes_).index(pred)
            st_shap = shap.plots.text(shap_values[0, :, target_class_idx], display=False)
            st.write(st_shap)
        except Exception as e:
            st.error(f"Could not compute SHAP explanation: {e}")

st.markdown("---")
st.caption("Trained with TF-IDF + Logistic Regression on Resume.csv")
