import streamlit as st
import pandas as pd
from pathlib import Path
from joblib import load
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge

st.set_page_config(page_title="AI Student Assistant — Frontend", layout="centered")
st.title("🎓 AI Student Assistant — Satisfaction Predictor (Frontend)")

project_dir = Path(__file__).resolve().parent
data_path = project_dir / "data" / "ai_student_sessions.csv"
model_path = project_dir / "models" / "ridge_model.joblib"

st.markdown("Predict a student's **satisfaction rating (1–5)** from session details.")

with st.sidebar:
    st.header("⚙️ Settings")
    if st.button("Re-train Ridge model"):
        if not data_path.exists():
            st.error("Dataset not found at data/ai_student_sessions.csv")
        else:
            df = pd.read_csv(data_path, parse_dates=["date"])
            X = df.drop(columns=["satisfaction_rating","date","session_id"])
            y = df["satisfaction_rating"]
            numeric_features = ["duration_min","num_prompts","ai_help_level","prompts_per_min","is_success"]
            categorical_features = ["education_level","subject","task_type","result","reuse_ai_again"]
            numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),("scaler", StandardScaler())])
            categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),("onehot", OneHotEncoder(handle_unknown="ignore"))])
            preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, numeric_features),("cat", categorical_transformer, categorical_features)])
            model = Pipeline(steps=[("preprocess", preprocessor),("model", Ridge(alpha=1.0, random_state=42))])
            model.fit(X, y)
            from joblib import dump
            dump(model, model_path)
            st.success("Model re-trained and saved.")

# Load or train model
if model_path.exists():
    model = load(model_path)
else:
    if not data_path.exists():
        st.error("Dataset missing. Place ai_student_sessions.csv into data/")
        st.stop()
    df = pd.read_csv(data_path, parse_dates=["date"])
    X = df.drop(columns=["satisfaction_rating","date","session_id"])
    y = df["satisfaction_rating"]
    numeric_features = ["duration_min","num_prompts","ai_help_level","prompts_per_min","is_success"]
    categorical_features = ["education_level","subject","task_type","result","reuse_ai_again"]
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),("onehot", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, numeric_features),("cat", categorical_transformer, categorical_features)])
    model = Pipeline(steps=[("preprocess", preprocessor),("model", Ridge(alpha=1.0, random_state=42))])
    model.fit(X, y)

st.header("🧩 Enter Session Details")
col1, col2 = st.columns(2)

with col1:
    education_level = st.selectbox("Education Level", ["High School","Undergraduate","Graduate"])
    subject = st.selectbox("Subject", ["Computer Science","Psychology","Business","Biology","Mathematics","History","Engineering"])
    task_type = st.selectbox("Task Type", ["Studying","Coding","Writing","Brainstorming","Homework Help"])
    result = st.selectbox("Session Result", ["Completed","Partial Complete","Drafted Ideas","Confused","Gave Up"])

with col2:
    duration_min = st.number_input("Session Duration (minutes)", min_value=10, max_value=240, value=60)
    num_prompts = st.number_input("Number of AI Prompts Used", min_value=1, max_value=60, value=10)
    ai_help_level = st.slider("AI Help Level (2–5)", 2, 5, 3)
    reuse_ai_again = st.radio("Would Use AI Again?", [True, False])

prompts_per_min = num_prompts / duration_min
is_success = 1 if result in ["Completed","Partial Complete"] else 0

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

if st.button("🔍 Predict Satisfaction"):
    pred = float(model.predict(input_df)[0])
    st.success(f"🎯 Predicted Satisfaction Rating: **{round(pred,2)} / 5**")
    if pred >= 4:
        st.balloons()

st.divider()
st.header("📊 Project Figures")
fig_dir = project_dir / "figures"
colA, colB, colC = st.columns(3)
with colA:
    st.image(str(fig_dir / "satisfaction_distribution.png"), caption="Satisfaction Rating Distribution")
with colB:
    st.image(str(fig_dir / "correlation_heatmap.png"), caption="Correlation Heatmap")
with colC:
    st.image(str(fig_dir / "pred_vs_actual_ridge.png"), caption="Predicted vs Actual (Ridge)")

st.info("Tip: Use the sidebar to retrain and update the model file.")