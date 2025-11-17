# app.py (robust, cloud-safe version)
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="AI Student Assistant — Frontend", layout="centered")
st.title("🎓 AI Student Assistant — Satisfaction Predictor (Frontend)")
st.markdown("Predict a student's **satisfaction rating (1–5)** from session details.")

# project paths (relative)
project_dir = Path(__file__).resolve().parent
data_path = project_dir / "data" / "ai_student_sessions.csv"
model_path = project_dir / "models" / "ridge_model.joblib"
fig_dir = project_dir / "figures"

# --- robust loader: try joblib, fallback to pickle ---
try:
    from joblib import load, dump
    _loader_name = "joblib"
except Exception:
    import pickle
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    def dump(obj, path):
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    _loader_name = "pickle (fallback)"

# --- ML imports (kept local to avoid import errors at top) ---
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge

# Feature lists (must match training)
numeric_features = ["duration_min", "num_prompts", "ai_help_level", "prompts_per_min", "is_success"]
categorical_features = ["education_level", "subject", "task_type", "result", "reuse_ai_again"]

def build_preprocessor():
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])
    return preprocessor

def build_model_pipeline():
    preprocessor = build_preprocessor()
    model = Pipeline(steps=[("preprocess", preprocessor), ("model", Ridge(alpha=1.0, random_state=42))])
    return model

# --- helper: train and save model using local CSV ---
def train_and_save_model():
    if not data_path.exists():
        st.error(f"Cannot train: dataset not found at {data_path}")
        return None
    df = pd.read_csv(data_path, parse_dates=["date"])
    X = df.drop(columns=["satisfaction_rating", "date", "session_id"])
    y = df["satisfaction_rating"]
    model = build_model_pipeline()
    model.fit(X, y)
    # ensure models dir exists
    model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, model_path)
    st.success("Model trained and saved.")
    st.info(f"Model saved using {_loader_name} at: {model_path.name}")
    return model

# Sidebar actions
with st.sidebar:
    st.header("⚙️ Settings")
    if st.button("Re-train Ridge model"):
        mdl = train_and_save_model()
        if mdl is not None:
            st.experimental_rerun()

# Load model (try to load saved model, otherwise train)
model = None
if model_path.exists():
    try:
        model = load(model_path)
        st.success(f"Loaded model from {model_path.name} using {_loader_name}.")
    except Exception as e:
        st.warning("Saved model could not be loaded (version mismatch or corruption). Retraining a new model in this environment...")
        st.write(f"Load error: {e}")
        model = train_and_save_model()
else:
    st.info("No saved model found — training a new Ridge model with local dataset (if available).")
    model = train_and_save_model()

if model is None:
    st.error("Model not available. Please ensure dataset exists and retrain from the sidebar.")
    st.stop()

# --- User input UI ---
st.header("🧩 Enter Session Details")
col1, col2 = st.columns(2)

with col1:
    education_level = st.selectbox("Education Level", ["High School", "Undergraduate", "Graduate"])
    subject = st.selectbox("Subject", ["Computer Science", "Psychology", "Business", "Biology", "Mathematics", "History", "Engineering"])
    task_type = st.selectbox("Task Type", ["Studying", "Coding", "Writing", "Brainstorming", "Homework Help"])
    result = st.selectbox("Session Result", ["Completed", "Partial Complete", "Drafted Ideas", "Confused", "Gave Up"])

with col2:
    duration_min = st.number_input("Session Duration (minutes)", min_value=10, max_value=240, value=60)
    num_prompts = st.number_input("Number of AI Prompts Used", min_value=1, max_value=60, value=10)
    ai_help_level = st.slider("AI Help Level (2–5)", 2, 5, 3)
    reuse_ai_again = st.radio("Would Use AI Again?", [True, False])

prompts_per_min = float(num_prompts) / float(max(duration_min, 1))
is_success = 1 if result in ["Completed", "Partial Complete"] else 0

input_df = pd.DataFrame([{
    "education_level": education_level,
    "subject": subject,
    "task_type": task_type,
    "duration_min": duration_min,
    "num_prompts": num_prompts,
    "ai_help_level": ai_help_level,
    "result": result,
    "reuse_ai_again": reuse_ai_again,
    "prompts_per_min": prompts_per_min,
    "is_success": is_success
}])

# Prediction action
if st.button("🔍 Predict Satisfaction"):
    try:
        raw_pred = float(model.predict(input_df)[0])
    except Exception as e:
        st.error("Prediction failed. This often indicates a mismatch between training features and input schema.")
        st.write("Error:", e)
        st.stop()

    # Clip & round for user-friendly display
    import numpy as np
    calibrated = float(np.clip(raw_pred, 1.0, 5.0))
    rounded = round(calibrated, 2)
    rounded_int = int(round(calibrated))  # for optional integer display

    st.success(f"🎯 Predicted Satisfaction Rating: **{rounded} / 5**  (rounded to nearest: {rounded_int})")
    st.write(f"Raw model output: {raw_pred:.4f}")
    if rounded >= 4.0:
        st.balloons()
    # show simple explanation of features used
    with st.expander("Show input used for prediction"):
        st.table(input_df.T)

# Figures and diagnostics
st.divider()
st.header("📊 Project Figures")
colA, colB, colC = st.columns(3)
with colA:
    if (fig_dir / "satisfaction_distribution.png").exists():
        st.image(str(fig_dir / "satisfaction_distribution.png"), caption="Satisfaction Rating Distribution")
    else:
        st.write("satisfaction_distribution.png not found")
with colB:
    if (fig_dir / "correlation_heatmap.png").exists():
        st.image(str(fig_dir / "correlation_heatmap.png"), caption="Correlation Heatmap")
    else:
        st.write("correlation_heatmap.png not found")
with colC:
    if (fig_dir / "pred_vs_actual_ridge.png").exists():
        st.image(str(fig_dir / "pred_vs_actual_ridge.png"), caption="Predicted vs Actual (Ridge)")
    else:
        st.write("pred_vs_actual_ridge.png not found")

st.info("Tip: Use the sidebar to retrain and update the model file. If deploying to Streamlit Cloud, include a requirements.txt with joblib and scikit-learn.")
