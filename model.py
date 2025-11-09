import pandas as pd
import sqlite3
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# --------------------------------------------------
# STEP 1: Load credit dataset from database.
# --------------------------------------------------
print("[INFO]     Loading dataset 'german_data' from 'credit_risk.db'...")
conn = sqlite3.connect("credit_risk.db")
df = pd.read_sql("SELECT * FROM german_data", conn)
conn.close()
print(f"[SUCCESS]  Dataset loaded: {len(df)} rows, {len(df.columns)} columns.\n")

# --------------------------------------------------
# STEP 2: Split dataset into features and target variable.
# --------------------------------------------------
print("[INFO]     Splitting dataset into features and target variable...")
X = df.drop("credit_risk", axis=1)
y = df["credit_risk"]

print("[INFO]     Splitting dataset into training and test sets (80/20 split)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"[SUCCESS]  Training samples: {len(X_train)} | Test samples: {len(X_test)}\n")

# --------------------------------------------------
# STEP 3: Build pipeline: 
#           - Preprocessing: Standard Scaling, One Hot Encoding
#           - Model: Logistic Regression
# --------------------------------------------------
print("[INFO]     Building pipeline (preprocessing and logistic regression model)...")
num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X_train.select_dtypes(include=["object", "category"]).columns

# The list of categories is based on the documentation of the dataset.
preprocessor=ColumnTransformer([
    ("scaler", StandardScaler(), num_cols),
    ("encoder", OneHotEncoder(
        categories=[
            ["A11", "A12", "A13", "A14"],
            ["A30", "A31", "A32", "A33", "A34"],
            ["A40", "A41", "A42", "A43", "A44", "A45", "A46", "A47", "A48", "A49", "A410"],
            ["A61", "A62", "A63", "A64", "A65"],
            ["A71", "A72", "A73", "A74", "A75"],
            ["A91", "A92", "A93", "A94", "A95"],
            ["A101", "A102", "A103"],
            ["A121", "A122", "A123", "A124"],
            ["A141", "A142", "A143"],
            ["A151", "A152", "A153"],
            ["A171", "A172", "A173", "A174"],
            ["A191", "A192"],
            ["A201", "A202"]
        ],        
        handle_unknown="ignore"
    ), 
    cat_cols)
])
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])
print("[SUCCESS]  Pipeline built.\n")

# --------------------------------------------------
# STEP 4: Fit pipeline.
# --------------------------------------------------
print("[INFO]     Fitting pipeline...")
pipeline.fit(X_train, y_train)
print("[SUCCESS]  Pipeline fitted.\n")

# --------------------------------------------------
# STEP 5: Save pipeline.
# --------------------------------------------------
print("[INFO]     Saving pipeline...")
joblib.dump(pipeline, "pipeline.joblib")
print("[SUCCESS]  Pipeline saved as 'pipeline.joblib'.\n")

# --------------------------------------------------
# STEP 6: Evaluate model performance on test set.
# --------------------------------------------------
print("[INFO]     Evaluating model on the test set...")
y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"[RESULT]   Model Accuracy: {accuracy:.4f}")
print("[RESULT]   Confusion Matrix (rows = actual, columns = predicted):\n")
print(pd.DataFrame(
    cm, 
    index=["Actual Bad", "Actual Good"], 
    columns=["Predicted Bad", "Predicted Good"]
    )
)
