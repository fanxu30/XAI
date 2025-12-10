import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Resume Categorizer", layout="wide")
st.title("Resume Categorization Model")
st.caption("This resume categorization model is trained on a dataset of resumes and predicts the category of a resume based on the text of the resume.")

# Load model - use relative path for deployment compatibility
model_path = Path(__file__).parent / "resume_model.joblib"
model = joblib.load(model_path)
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
        st.markdown("Words highlighted in **green** push the prediction toward this category. Words in **red** push away from it.")
        try:
            shap_values = explainer([resume_text])
            target_class_idx = list(model.classes_).index(pred)
            
            # Get tokens and their SHAP values
            tokens = shap_values.data[0]
            values = shap_values.values[0, :, target_class_idx]
            
            # Normalize values for coloring
            max_abs = max(1e-8, float(np.abs(values).max()))
            
            # Build highlighted HTML
            html_parts = []
            for tok, val in zip(tokens, values):
                norm = val / max_abs  # -1 to 1
                # Green for positive, red for negative
                if norm >= 0:
                    # Green with intensity based on value
                    intensity = min(1.0, abs(norm))
                    bg_color = f"rgba(0, 200, 0, {intensity * 0.7})"
                else:
                    # Red with intensity based on value
                    intensity = min(1.0, abs(norm))
                    bg_color = f"rgba(255, 0, 0, {intensity * 0.7})"
                
                html_parts.append(
                    f'<span style="background:{bg_color}; padding:2px 4px; margin:1px; '
                    f'border-radius:3px; display:inline-block; line-height:1.8; font-size:14px;">{tok}</span>'
                )
            
            highlighted_html = " ".join(html_parts)
            
            full_html = f"""
            <div style="background:#f9f9f9; color:#222; padding:15px; border-radius:8px; 
                        border:1px solid #ddd; font-family:Arial, sans-serif; line-height:2;">
                {highlighted_html}
            </div>
            """
            
            components.html(full_html, height=500, scrolling=True)
            
            # Show top contributing words with custom matplotlib bar chart
            st.markdown("### Top Contributing Words")
            
            # Get top N tokens by absolute SHAP value
            token_importance = list(zip(tokens, values))
            token_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            top_n = 20 
            
            # Extract top tokens and values
            top_tokens = [t for t, v in token_importance[:top_n]]
            top_values = [v for t, v in token_importance[:top_n]]
            
            # Sort by value for better visualization
            sorted_indices = np.argsort(top_values)
            sorted_tokens = [top_tokens[i] for i in sorted_indices]
            sorted_values = [top_values[i] for i in sorted_indices]
            
            # Create custom matplotlib bar chart
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create colors: red for negative, blue for positive
            colors = ['#ff0051' if v < 0 else '#0080ff' for v in sorted_values]
            
            # Create horizontal bar plot
            y_pos = np.arange(len(sorted_tokens))
            bars = ax.barh(y_pos, sorted_values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Customize the plot
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_tokens, fontsize=10)
            ax.set_xlabel('SHAP Value', fontsize=12, fontweight='bold')
            ax.set_title(f'Top {top_n} Contributing Words for "{pred}" Category', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
            
            # Calculate x-axis limits with padding for labels
            x_min = min(sorted_values)
            x_max = max(sorted_values)
            x_range = x_max - x_min
            x_range_abs = max(abs(x_min), abs(x_max))
            # Increase padding significantly to prevent overlap
            padding = max(0.15 * abs(x_min), 0.15 * abs(x_max), 0.25 * x_range, 0.15 * x_range_abs)
            
            # Set x-axis limits to accommodate labels
            ax.set_xlim(x_min - padding, x_max + padding)
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, sorted_values)):
                width = bar.get_width()
                bar_width_abs = abs(width)
                
                if bar_width_abs > 0.08 * x_range_abs:
                    # Label inside bar
                    label_x = width / 2
                    text_color = 'white'
                    ha = 'center'
                else:
                    # Label outside bar
                    label_x = width + (0.05 * x_range_abs if width >= 0 
                                      else width - (0.05 * x_range_abs))
                    text_color = 'black'
                    ha = 'left' if width >= 0 else 'right'
                
                ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                       f'{val:.4f}', ha=ha, 
                       va='center', fontsize=9, fontweight='bold', color=text_color)
            
            # Format x-axis to avoid scientific notation
            ax.xaxis.set_major_formatter(plt.FuncFormatter(
                lambda x, p: f'{x:.4f}' if abs(x) < 1 else f'{x:.3f}' if abs(x) < 100 else f'{x:.2e}'
            ))
            
            # Invert y-axis so highest values are at top
            ax.invert_yaxis()
            
            # Adjust layout with more bottom padding for x-axis labels
            # Increase bottom margin to give more space for x-axis labels
            plt.tight_layout(rect=[0, 0.08, 1, 0.95])

            plt.subplots_adjust(bottom=0.12)
            st.pyplot(fig)
            plt.close(fig)
                    
        except Exception as e:
            st.error(f"Could not compute SHAP explanation: {e}")

st.markdown("---")
st.caption("Models used were TF-IDF + Logistic Regression")