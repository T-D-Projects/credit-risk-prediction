# credit-risk-prediction

## Description

Predicts credit risk using a logistic regression model trained on the "German Credit Data" dataset.

## Dataset

This project uses the ["German Credit Data"](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) dataset. 

**Important:** Please download the ZIP file and extract `german.data` into the project folder before running the scripts.

## Usage

The project contains three scripts:

- **`datasets.py`**  
  - Loads the "German Credit Data" dataset `german.data`.  
  - Performs basic preprocessing:
    - Assigns column names according to the dataset documentation `german.doc` included in the ZIP file. 
    - Converts the target variable `credit_risk` (1 = Good, 0 = Bad).  
  - Creates the database `credit_risk.db` and saves the first 990 rows as table `german_data`. 
  - Saves the remaining 10 rows without the target variable as `credit_applications.csv`. 

- **`model.py`**  
  - Loads the credit dataset from the database.
  - Splits the dataset into training and test sets.
  - Builds a pipeline consisting of:
    - Preprocessing: Standard Scaling, One Hot Encoding
    - Model: Logistic Regression
  - Fits the pipeline on the training set and saves it as `pipeline.joblib`.
  - Evaluates the model on the test set.

- **`prediction.py`**  
  - Loads the credit applications and the pipeline.  
  - Generates predictions for the credit applications.  
  - Saves the credit applications and the predictions as `predictions.csv`.

## Requirements

- `pandas`
- `scikit-learn`
- `joblib`
