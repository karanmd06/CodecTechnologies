"""
Fraud Detection System - Complete Python script (Jupyter-friendly)

Instructions:
- Place your dataset CSV in `data/creditcard.csv` or change DATA_PATH below.
- Recommended packages: pandas, numpy, scikit-learn, imbalanced-learn, matplotlib, seaborn, joblib, tensorflow (for autoencoder).
  Install with: pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib tensorflow

What this script does:
1. Loads dataset
2. Cleans and preprocesses data (scales Time and Amount)
3. Trains unsupervised Isolation Forest and supervised models (Logistic Regression, Random Forest)
4. Builds and trains an Autoencoder for anomaly detection
5. Handles class imbalance with SMOTE for supervised models
6. Evaluates models with Precision/Recall/F1 and ROC-AUC
7. Saves best model(s) to disk and provides prediction helpers

Run cell-by-cell if using a notebook, or run as a script.
"""

# --- Imports ----------------------------------------------------------------
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import matplotlib.pyplot as plt
import seaborn as sns

import joblib

# TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# --- Settings ----------------------------------------------------------------
DATA_PATH = 'data/creditcard.csv'  # change if needed
RANDOM_STATE = 42
TEST_SIZE = 0.25

# --- Load dataset -----------------------------------------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please put the CSV at this path or change DATA_PATH.")

df = pd.read_csv(DATA_PATH)
print('Dataset loaded. Shape:', df.shape)

# --- Quick EDA --------------------------------------------------------------
print('\n--- Head ---')
print(df.head())

print('\n--- Info ---')
print(df.info())

print('\n--- Target distribution ---')
print(df['Class'].value_counts())

# --- Preprocessing ---------------------------------------------------------
# Features: V1..V28 (already scaled), Time, Amount
# We'll scale Time and Amount
feature_cols = [c for c in df.columns if c != 'Class']

scaler = StandardScaler()
df[['ScaledTime', 'ScaledAmount']] = scaler.fit_transform(df[['Time', 'Amount']])

# Create final features list
X = df.drop(['Time', 'Amount', 'ScaledTime', 'ScaledAmount', 'Class'], axis=1)
# Insert scaled columns
X['ScaledTime'] = df['ScaledTime']
X['ScaledAmount'] = df['ScaledAmount']
# ensure column order
X = X[[c for c in df.columns if c.startswith('V')] + ['ScaledTime', 'ScaledAmount']]

y = df['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
print('\nTrain shape:', X_train.shape, 'Test shape:', X_test.shape)
print('Train fraud ratio:', y_train.mean(), 'Test fraud ratio:', y_test.mean())

# --- Unsupervised: Isolation Forest ----------------------------------------
print('\nTraining Isolation Forest (unsupervised anomaly detection)...')
# Fit on training data (we can fit on full training set or only non-fraud samples). Typical approach: fit on majority class to learn normal.
X_train_nonfraud = X_train[y_train == 0]

iso = IsolationForest(n_estimators=200, contamination=y_train.mean(), random_state=RANDOM_STATE)
iso.fit(X_train_nonfraud)

# Predict anomalies on test set: -1 for anomaly (fraud), 1 for normal
iso_pred = iso.predict(X_test)
# convert to 1 for fraud, 0 for normal
iso_pred_labels = np.where(iso_pred == -1, 1, 0)

print('\nIsolation Forest evaluation:')
print(classification_report(y_test, iso_pred_labels, digits=4))
try:
    iso_scores = -iso.decision_function(X_test)  # higher -> more anomalous
    iso_auc = roc_auc_score(y_test, iso_scores)
    print('Isolation Forest ROC AUC:', iso_auc)
except Exception:
    print('Could not compute ROC AUC for Isolation Forest')

cm = confusion_matrix(y_test, iso_pred_labels)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix - Isolation Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# --- Supervised models with SMOTE -----------------------------------------
print('\nPreparing supervised models with SMOTE...')
smote = SMOTE(random_state=RANDOM_STATE)
X_res, y_res = smote.fit_resample(X_train, y_train)
print('After SMOTE, class distribution:', np.bincount(y_res))

# Logistic Regression
print('\nTraining Logistic Regression...')
log_clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
log_clf.fit(X_res, y_res)
log_pred = log_clf.predict(X_test)
log_proba = log_clf.predict_proba(X_test)[:,1]

print('Logistic Regression evaluation:')
print(classification_report(y_test, log_pred, digits=4))
print('Logistic ROC AUC:', roc_auc_score(y_test, log_proba))

# Random Forest
print('\nTraining Random Forest...')
rf_clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=RANDOM_STATE)
rf_clf.fit(X_res, y_res)
rf_pred = rf_clf.predict(X_test)
rf_proba = rf_clf.predict_proba(X_test)[:,1]

print('Random Forest evaluation:')
print(classification_report(y_test, rf_pred, digits=4))
print('Random Forest ROC AUC:', roc_auc_score(y_test, rf_proba))

# Confusion matrices
for name, pred in [('Logistic Regression', log_pred), ('Random Forest', rf_pred)]:
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# --- Autoencoder (if TensorFlow available) --------------------------------
ae_model = None
ae_threshold = None
if TF_AVAILABLE:
    print('\nTensorFlow available — building Autoencoder...')
    # Train autoencoder on non-fraud training samples
    X_train_ae = X_train[y_train == 0].values

    input_dim = X_train_ae.shape[1]
    encoding_dim = int(input_dim / 2)

    ae = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(encoding_dim, activation='relu'),
        layers.Dense(int(encoding_dim/2), activation='relu'),
        layers.Dense(encoding_dim, activation='relu'),
        layers.Dense(input_dim, activation='linear')
    ])

    ae.compile(optimizer='adam', loss='mse')
    history = ae.fit(X_train_ae, X_train_ae,
                     epochs=50,
                     batch_size=256,
                     validation_split=0.1,
                     callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
                     verbose=1)

    # reconstruction error on test set
    reconstructions = ae.predict(X_test.values)
    mse = np.mean(np.power(X_test.values - reconstructions, 2), axis=1)

    # choose threshold: mean + 3*std of reconstruction error on non-fraud test samples
    recon_nonfraud = mse[y_test == 0]
    ae_threshold = recon_nonfraud.mean() + 3 * recon_nonfraud.std()
    ae_pred_labels = (mse > ae_threshold).astype(int)

    print('\nAutoencoder evaluation:')
    print(classification_report(y_test, ae_pred_labels, digits=4))
    try:
        ae_auc = roc_auc_score(y_test, mse)
        print('Autoencoder ROC AUC:', ae_auc)
    except Exception:
        print('Could not compute ROC AUC for Autoencoder')

    cm = confusion_matrix(y_test, ae_pred_labels)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix - Autoencoder')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    ae_model = ae
else:
    print('\nTensorFlow not available — skipping Autoencoder. To enable, install tensorflow.')

# --- Model comparison -----------------------------------------------------
print('\nComparing model F1 scores and AUCs...')
results = {}
results['IsolationForest'] = {
    'f1': f1_score(y_test, iso_pred_labels),
    'auc': iso_auc if 'iso_auc' in locals() else None
}
results['LogisticRegression'] = {
    'f1': f1_score(y_test, log_pred),
    'auc': roc_auc_score(y_test, log_proba)
}
results['RandomForest'] = {
    'f1': f1_score(y_test, rf_pred),
    'auc': roc_auc_score(y_test, rf_proba)
}
if ae_model is not None:
    results['Autoencoder'] = {
        'f1': f1_score(y_test, ae_pred_labels),
        'auc': ae_auc if 'ae_auc' in locals() else None
    }

for k,v in results.items():
    print(k, v)

# Choose best supervised model by AUC (or F1 if you prefer)
best_model_name = None
best_auc = -1
for name in ['RandomForest', 'LogisticRegression']:
    auc = results[name]['auc']
    if auc is not None and auc > best_auc:
        best_auc = auc
        best_model_name = name

print(f"\nBest supervised model by AUC: {best_model_name} (AUC={best_auc})")

# Save models
os.makedirs('models', exist_ok=True)
joblib.dump(iso, 'models/isolation_forest.joblib')
joblib.dump(log_clf, 'models/logistic_regression.joblib')
joblib.dump(rf_clf, 'models/random_forest.joblib')
print('Saved Isolation Forest, Logistic Regression, and Random Forest to models/ folder')
if ae_model is not None:
    ae_model.save('models/autoencoder')
    print('Saved Autoencoder to models/autoencoder')

# --- Helper prediction functions ------------------------------------------

def predict_supervised(sample_df, model_path='models/random_forest.joblib'):
    """Predict using a supervised saved model. sample_df should be a pandas DataFrame with same feature columns as X."""
    model = joblib.load(model_path)
    proba = model.predict_proba(sample_df)[:,1]
    pred = (proba >= 0.5).astype(int)
    return pred, proba


def predict_isolation(sample_df, model_path='models/isolation_forest.joblib'):
    """Predict anomalies using Isolation Forest. Returns 1 for fraud/anomaly, 0 for normal."""
    model = joblib.load(model_path)
    pred = model.predict(sample_df)
    labels = np.where(pred == -1, 1, 0)
    try:
        scores = -model.decision_function(sample_df)
    except Exception:
        scores = None
    return labels, scores


def predict_autoencoder(sample_df, model_path='models/autoencoder', threshold=None):
    """Predict anomalies using saved autoencoder. If threshold is None, use computed ae_threshold from training.
    Returns 1 for fraud, 0 for normal, and reconstruction error.
    """
    if not TF_AVAILABLE:
        raise RuntimeError('TensorFlow not available')
    model = keras.models.load_model(model_path)
    recon = model.predict(sample_df.values)
    mse = np.mean(np.power(sample_df.values - recon, 2), axis=1)
    thr = threshold if threshold is not None else ae_threshold
    labels = (mse > thr).astype(int)
    return labels, mse

print('\nScript finished. Models and helpers are saved in the models/ directory.')

# Example usage (uncomment and modify to test):
# sample = X_test.iloc[[0]]
# print('Supervised prediction (RF):', predict_supervised(sample))
# print('IsolationForest prediction:', predict_isolation(sample))
# if TF_AVAILABLE:
#     print('Autoencoder prediction:', predict_autoencoder(sample))
