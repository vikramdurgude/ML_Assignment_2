import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)
from model.models import load_models, MODEL_FILES

st.set_page_config(page_title="WDBC ML Classifier", page_icon="🧬", layout="wide")

FEATURES = [
    "mean_radius","mean_texture","mean_perimeter","mean_area","mean_smoothness",
    "mean_compactness","mean_concavity","mean_concave_points","mean_symmetry","mean_fractal_dimension",
    "radius_se","texture_se","perimeter_se","area_se","smoothness_se",
    "compactness_se","concavity_se","concave_points_se","symmetry_se","fractal_dimension_se",
    "worst_radius","worst_texture","worst_perimeter","worst_area","worst_smoothness",
    "worst_compactness","worst_concavity","worst_concave_points","worst_symmetry","worst_fractal_dimension"
]

@st.cache_resource
def get_models():
    return load_models()

st.title("🧬 Wisconsin Diagnostic Breast Cancer — ML Classifier")
st.write(
    "Interactive comparison of six classification models trained on the "
    "Wisconsin Diagnostic Breast Cancer (WDBC) dataset."
)

models = get_models()

with st.sidebar:
    st.header("Model Selection")
    selected = st.selectbox("Choose a model", list(models.keys()))
    st.markdown("---")
    st.caption("The supplied test_data.csv is the evaluation dataset.")

uploaded = st.file_uploader("Upload test data CSV", type=["csv"])

if uploaded is not None:
    data = pd.read_csv(uploaded)
    st.success(f"Loaded {len(data)} rows.")
else:
    default_path = Path("test_data.csv")
    if default_path.exists():
        data = pd.read_csv(default_path)
        st.info("Using the included test_data.csv.")
    else:
        data = None

if data is not None:
    missing = [c for c in FEATURES if c not in data.columns]
    if missing:
        st.error("Missing required feature columns: " + ", ".join(missing))
        st.stop()

    X = data[FEATURES]
    model = models[selected]
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]

    st.subheader(f"Results — {selected}")

    if "target" in data.columns:
        y = data["target"].astype(int)
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{accuracy_score(y, pred):.4f}")
        c2.metric("AUC", f"{roc_auc_score(y, prob):.4f}")
        c3.metric("Precision", f"{precision_score(y, pred):.4f}")

        c4, c5, c6 = st.columns(3)
        c4.metric("Recall", f"{recall_score(y, pred):.4f}")
        c5.metric("F1 Score", f"{f1_score(y, pred):.4f}")
        c6.metric("MCC", f"{matthews_corrcoef(y, pred):.4f}")

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y, pred)
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Benign (0)", "Actual Malignant (1)"],
            columns=["Predicted Benign (0)", "Predicted Malignant (1)"]
        )
        st.dataframe(cm_df, use_container_width=True)

        st.subheader("Classification Report")
        report = classification_report(
            y, pred, target_names=["Benign", "Malignant"], output_dict=True
        )
        st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

    st.subheader("Predictions")
    result = data.copy()
    result["predicted_target"] = pred
    result["malignant_probability"] = np.round(prob, 4)
    st.dataframe(result, use_container_width=True)

    st.download_button(
        "Download Predictions CSV",
        result.to_csv(index=False).encode("utf-8"),
        "predictions.csv",
        "text/csv"
    )
else:
    st.warning("Upload test_data.csv to view model predictions and metrics.")
