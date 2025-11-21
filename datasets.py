import os
import pandas as pd
import sqlite3

# --------------------------------------------------
# STEP 1: Load credit dataset.
# --------------------------------------------------
print("[INFO]     Loading dataset 'german.data'...")
folder_path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(folder_path, "german.data")
df = pd.read_csv(path, sep=r"\s+", header=None)
print(f"[SUCCESS]  Dataset loaded with {len(df)} rows and {len(df.columns)} columns.\n")

# --------------------------------------------------
# STEP 2: Assign column names according to dataset documentation.
# --------------------------------------------------
print("[INFO]     Assigning column names...")
df.columns = [
    'checking_account', 'duration', 'credit_history', 'purpose', 'credit_amount',
    'savings_account', 'employment_since', 'installment_rate', 'personal_status',
    'guarantors', 'residence_since', 'property', 'age', 'other_installment_plans',
    'housing', 'number_existing_credits', 'job', 'liable_maintenance_people',
    'telephone', 'foreign_worker', 'credit_risk'
]
print("[SUCCESS]  Column names assigned.\n")

# --------------------------------------------------
# STEP 3: Convert target variable (1 = Good, 0 = Bad).
# --------------------------------------------------
print("[INFO]     Converting target variable 'credit_risk' (1 = Good, 0 = Bad)...")
df['credit_risk'] = df['credit_risk'].replace({1: 1, 2: 0})
print("[SUCCESS]  Target variable conversion completed.\n")

# --------------------------------------------------
# STEP 4: Create SQLite database and save first 990 rows.
# --------------------------------------------------
print("[INFO]     Creating a SQLite database and saving the first 990 rows to it...")
path = os.path.join(folder_path, "credit_risk.db")
conn = sqlite3.connect(path)
df.head(990).to_sql("german_data", conn, if_exists="replace", index=False)
conn.close()
print("[SUCCESS]  Dataset stored in table 'german_data' of database 'credit_risk.db'.\n")

# --------------------------------------------------
# STEP 5: Export last 10 rows (excluding target) to CSV.
# --------------------------------------------------
print("[INFO]     Exporting the last 10 rows (excluding the target) to CSV...")
df_no_target = df.drop("credit_risk", axis=1)
path = os.path.join(folder_path, "credit_applications.csv")
df_no_target.tail(10).to_csv(path, index=False)
print("[SUCCESS]  Export complete: 'credit_applications.csv'.\n")
