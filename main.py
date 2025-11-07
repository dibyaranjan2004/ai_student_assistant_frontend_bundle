from joblib import dump
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge

data_path = Path("data/ai_student_sessions.csv")
df = pd.read_csv(data_path)

X = df.drop(columns=["satisfaction_rating", "date", "session_id"])
y = df["satisfaction_rating"]

numeric_features = ["duration_min","num_prompts","ai_help_level","prompts_per_min","is_success"]
categorical_features = ["education_level","subject","task_type","result","reuse_ai_again"]

numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

model = Pipeline(steps=[("preprocess", preprocessor), ("model", Ridge(alpha=1.0, random_state=42))])
model.fit(X, y)
dump(model, Path("models/ridge_model.joblib"))
print("✅ Model retrained and saved successfully.")
