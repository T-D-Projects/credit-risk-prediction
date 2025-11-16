import pandas as pd
import joblib

# --------------------------------------------------
# STEP 1: Load credit applications dataset.
# --------------------------------------------------
print("[INFO]     Loading dataset 'credit_applications.csv'...")
df = pd.read_csv("credit_applications.csv")
print(f"[SUCCESS]  Dataset loaded: {len(df)} rows, {len(df.columns)} columns.\n")

# --------------------------------------------------
# STEP 2: Load fitted pipeline (preprocessing and logistic regression model).
# --------------------------------------------------
print("[INFO]     Loading pipeline 'pipeline.joblib' (preprocessing and logistic regression model)...")
pipeline = joblib.load("pipeline.joblib")
print("[SUCCESS]  Pipeline loaded.\n")

# --------------------------------------------------
# STEP 3: Generate credit risk predictions.
# --------------------------------------------------
print("[INFO]     Generating credit risk predictions...")
df_pred = pipeline.predict(df)
df["predicted_credit_risk"] = df_pred
print("[SUCCESS]  Predictions generated.\n")

# --------------------------------------------------
# STEP 4: Save original dataset and predictions to a single CSV.
# --------------------------------------------------
print("[INFO]     Saving predictions...")
df.to_csv("predictions.csv", index=False)
print("[SUCCESS]  Credit applications and predictions saved as 'predictions.csv'.\n")
