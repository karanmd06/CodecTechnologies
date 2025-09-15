"""
Customer Churn Prediction - Complete Python script (Jupyter-friendly)

Instructions:
- Place your dataset CSV in `data/Telco-Customer-Churn.csv` or change the path below.
- Recommended packages: pandas, numpy, scikit-learn, imbalanced-learn, matplotlib, seaborn, joblib.
  Install with: pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib

What this script does:
1. Loads dataset
2. Cleans and preprocesses data (handles missing values, encodes categoricals, scales numerics)
3. Handles class imbalance with SMOTE
4. Trains Logistic Regression, Decision Tree, and Neural Network (MLPClassifier)
5. Evaluates models with classification report, confusion matrix, ROC-AUC
6. Saves the best model to disk
7. Provides a minimal example of a prediction function

Run cell-by-cell if using a notebook, or run as a script.
"""

# --- Imports ----------------------------------------------------------------
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import matplotlib.pyplot as plt
import seaborn as sns

import joblib

# --- Settings ----------------------------------------------------------------
DATA_PATH = 'data/Telco-Customer-Churn.csv'  # change if needed
RANDOM_STATE = 42
TEST_SIZE = 0.25

# --- Load dataset -----------------------------------------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please put the Telco CSV at this path or change DATA_PATH.")

raw = pd.read_csv(DATA_PATH)
print('Dataset loaded. Shape:', raw.shape)

# --- Quick EDA --------------------------------------------------------------
print('\n--- Head ---')
print(raw.head())

print('\n--- Info ---')
print(raw.info())

print('\n--- Target distribution ---')
print(raw['Churn'].value_counts(dropna=False))

# --- Basic cleaning --------------------------------------------------------
# Some Telco datasets have TotalCharges as object; convert
if 'TotalCharges' in raw.columns and raw['TotalCharges'].dtype == object:
    raw['TotalCharges'] = raw['TotalCharges'].replace(' ', np.nan)
    raw['TotalCharges'] = raw['TotalCharges'].astype(float)

# Drop customerID if present
if 'customerID' in raw.columns:
    raw = raw.drop('customerID', axis=1)

# Target
raw['Churn'] = raw['Churn'].map({'Yes': 1, 'No': 0})

# Identify feature types
num_cols = raw.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'Churn' in num_cols:
    num_cols.remove('Churn')
cat_cols = raw.select_dtypes(include=['object', 'category']).columns.tolist()

print('\nNumerical columns:', num_cols)
print('Categorical columns:', cat_cols)

# --- Train / Test split ----------------------------------------------------
X = raw.drop('Churn', axis=1)
y = raw['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
print('\nTrain shape:', X_train.shape, 'Test shape:', X_test.shape)

# --- Preprocessing pipelines -----------------------------------------------
# Numerical pipeline: impute (median) + scale
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline: impute (most frequent) + one-hot encode (drop='first' to avoid multicollinearity)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ]
)

# If there are no categorical columns (rare), handle accordingly
# --- Create imbalanced pipeline with SMOTE ---------------------------------
smote = SMOTE(random_state=RANDOM_STATE)

# We'll create a utility function to train and evaluate models

def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test, use_smote=True):
    """Train model in an imblearn pipeline (preprocessor -> SMOTE -> model) and evaluate."""
    steps = []
    steps.append(('preprocessor', preprocessor))
    if use_smote:
        # SMOTE works on numeric arrays, so must come after preprocessor
        steps.append(('smote', smote))
    steps.append(('clf', model))

    pipeline = ImbPipeline(steps=steps)

    print(f"\nTraining {model_name}...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = None
    if hasattr(pipeline, 'predict_proba') or hasattr(pipeline.named_steps['clf'], 'predict_proba'):
        try:
            y_proba = pipeline.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None

    print(f"\n--- Evaluation ({model_name}) ---")
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=4))

    if y_proba is not None:
        auc = roc_auc_score(y_test, y_proba)
        print('ROC AUC:', auc)

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})')
        plt.plot([0,1],[0,1],'--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc='lower right')
        plt.show()

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()

    return pipeline

# --- Models to train -------------------------------------------------------
# Logistic Regression
log_clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
# Decision Tree
dt_clf = DecisionTreeClassifier(random_state=RANDOM_STATE)
# Neural Network (MLP)
mlp_clf = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, random_state=RANDOM_STATE)

# Train & evaluate each
log_pipeline = train_and_evaluate(log_clf, 'Logistic Regression', X_train, X_test, y_train, y_test, use_smote=True)
dt_pipeline = train_and_evaluate(dt_clf, 'Decision Tree', X_train, X_test, y_train, y_test, use_smote=True)
mlp_pipeline = train_and_evaluate(mlp_clf, 'MLP Neural Network', X_train, X_test, y_train, y_test, use_smote=True)

# --- Compare AUCs (if available) -------------------------------------------
def get_proba_auc(pipeline, X_test, y_test):
    try:
        y_proba = pipeline.predict_proba(X_test)[:,1]
        return roc_auc_score(y_test, y_proba)
    except Exception:
        return None

results = {
    'Logistic Regression': get_proba_auc(log_pipeline, X_test, y_test),
    'Decision Tree': get_proba_auc(dt_pipeline, X_test, y_test),
    'MLP Neural Network': get_proba_auc(mlp_pipeline, X_test, y_test)
}

print('\nModel AUCs:')
for k,v in results.items():
    print(f"{k}: {v}")

# Choose best model by AUC (fallback to accuracy if AUC missing)
best_name = None
best_score = -1
for name, pipeline in [('Logistic Regression', log_pipeline), ('Decision Tree', dt_pipeline), ('MLP Neural Network', mlp_pipeline)]:
    score = get_proba_auc(pipeline, X_test, y_test)
    if score is None:
        # use accuracy
        score = accuracy_score(y_test, pipeline.predict(X_test))
    if score > best_score:
        best_score = score
        best_name = name

print(f"\nBest model: {best_name} (score: {best_score})")

# Save best pipeline to disk
model_map = {
    'Logistic Regression': log_pipeline,
    'Decision Tree': dt_pipeline,
    'MLP Neural Network': mlp_pipeline
}

best_pipeline = model_map[best_name]
joblib.dump(best_pipeline, 'best_churn_model.joblib')
print('Saved best model to best_churn_model.joblib')

# --- Example prediction function ------------------------------------------
def predict_single(sample_dict, model_path='best_churn_model.joblib'):
    """Predict churn probability for a single customer described by sample_dict.
    Example sample_dict keys should match the feature column names in the original dataset.
    """
    model = joblib.load(model_path)
    sample_df = pd.DataFrame([sample_dict])
    proba = None
    try:
        proba = model.predict_proba(sample_df)[:,1][0]
    except Exception:
        # if pipeline doesn't support predict_proba, return class
        pred = model.predict(sample_df)[0]
        return {'churn_pred': int(pred), 'churn_proba': None}
    return {'churn_pred': int(proba>=0.5), 'churn_proba': float(proba)}

# Example usage (replace keys/values with your dataset's features):
# sample = {
#     'gender': 'Male',
#     'SeniorCitizen': 0,
#     'Partner': 'Yes',
#     'Dependents': 'No',
#     'tenure': 12,
#     'PhoneService': 'Yes',
#     'MultipleLines': 'No',
#     'InternetService': 'DSL',
#     'OnlineSecurity': 'No',
#     'OnlineBackup': 'Yes',
#     'DeviceProtection': 'No',
#     'TechSupport': 'No',
#     'StreamingTV': 'No',
#     'StreamingMovies': 'No',
#     'Contract': 'Month-to-month',
#     'PaperlessBilling': 'Yes',
#     'PaymentMethod': 'Electronic check',
#     'MonthlyCharges': 70.35,
#     'TotalCharges': 1390.5
# }
# print(predict_single(sample))

print('\nScript finished.')
