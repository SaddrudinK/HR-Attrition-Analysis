"""
HR Attrition Analysis — analysis.py


Rough summary:
- Loads dataset (path from --data)
- Performs quick EDA and saves a few figures
- Preprocesses features (encoding + scaling)
- Handles class imbalance with SMOTE
- Trains Logistic Regression and Random Forest
- Evaluates models (precision, recall, f1, ROC AUC)
- Saves best model to outputs/models/best_model.joblib


Usage:
python analysis.py --data data/HR_comma_sep.csv --out outputs/


"""

import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import joblib

RANDOM_STATE = 42

def ensure_dirs(out_path):
(out_path / "figures").mkdir(parents=True, exist_ok=True)
(out_path / "models").mkdir(parents=True, exist_ok=True)

def load_data(path):
df = pd.read_csv(path)
return df

def quick_eda(df, out_path):
# Basic shape
print("Dataset shape:", df.shape)
# Target distribution
fig, ax = plt.subplots()
df['left'].value_counts().plot(kind='bar', ax=ax)
ax.set_title('Attrition distribution (0=stayed, 1=left)')
ax.set_xlabel('left')
fig.savefig(out_path / 'figures' / 'target_distribution.png')
plt.close(fig)

# Satisfaction vs left
fig, ax = plt.subplots()
sns.boxplot(x='left', y='satisfaction_level', data=df, ax=ax)
ax.set_title('Satisfaction by attrition')
fig.savefig(out_path / 'figures' / 'satisfaction_by_left.png')
plt.close(fig)


# Average monthly hours by left
fig, ax = plt.subplots()
sns.boxplot(x='left', y='average_montly_hours', data=df, ax=ax)
ax.set_title('Avg monthly hours by attrition')
fig.savefig(out_path / 'figures' / 'hours_by_left.png')
plt.close(fig)


# Left rate by department
dept = df.groupby('department')['left'].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8,4))
dept.plot(kind='bar', ax=ax)
ax.set_ylabel('Left rate')
ax.set_title('Attrition rate by department')
fig.savefig(out_path / 'figures' / 'attrition_by_department.png')
plt.close(fig)

def preprocess(df):
# Target
y = df['left']
X = df.drop(columns=['left'])


# Identify numerical and categorical columns
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()


# Some columns may be encoded as numbers but are actually categorical
# salary is categorical but stored as object in this dataset already


# Build preprocessing pipeline
num_pipeline = Pipeline([
("imputer", SimpleImputer(strategy='median')),
("scaler", StandardScaler())
])


cat_pipeline = Pipeline([
("imputer", SimpleImputer(strategy='most_frequent')),
("onehot", OneHotEncoder(handle_unknown='ignore', sparse=False))
])



preprocessor = ColumnTransformer([
("num", num_pipeline, num_cols),
("cat", cat_pipeline, cat_cols)
])


return X, y, preprocessor, num_cols, cat_cols




def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor, out_path):
# Resample (SMOTE) must be applied after preprocessing or on numeric features; easier approach
# is to preprocess, then SMOTE on numeric array. We'll create a pipeline that first transforms.


# Create pipelines for two models
lr_pipe = Pipeline([
("preproc", preprocessor),
("smote", SMOTE(random_state=RANDOM_STATE)),
("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
])



rf_pipe = Pipeline([
("preproc", preprocessor),
("smote", SMOTE(random_state=RANDOM_STATE)),
("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))
])


# Fit
print("Training Logistic Regression...")
lr_pipe.fit(X_train, y_train)
print("Training Random Forest...")
rf_pipe.fit(X_train, y_train)


# Predict
y_pred_lr = lr_pipe.predict(X_test)
y_pred_rf = rf_pipe.predict(X_test)


# Probabilities for ROC
y_prob_lr = lr_pipe.predict_proba(X_test)[:, 1]
y_prob_rf = rf_pipe.predict_proba(X_test)[:, 1]



# Evaluation prints
print('\nLogistic Regression classification report:')
print(classification_report(y_test, y_pred_lr))
print('\nRandom Forest classification report:')
print(classification_report(y_test, y_pred_rf))


# ROC AUC
lr_auc = roc_auc_score(y_test, y_prob_lr)
rf_auc = roc_auc_score(y_test, y_prob_rf)
print(f"Logistic ROC AUC: {lr_auc:.4f}")
print(f"RandomForest ROC AUC: {rf_auc:.4f}")


# Save confusion matrix figure for best model (choose by AUC)
best_pipe = rf_pipe if rf_auc >= lr_auc else lr_pipe
best_name = 'random_forest' if rf_auc >= lr_auc else 'logistic_regression'
cm = confusion_matrix(y_test, best_pipe.predict(X_test))
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title(f'Confusion Matrix - {best_name}')
fig.savefig(out_path / f'figures/confusion_matrix_{best_name}.png')
plt.close(fig)



# ROC curve
fpr, tpr, _ = roc_curve(y_test, best_pipe.predict_proba(X_test)[:,1])
fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.set_title(f'ROC Curve - {best_name} (AUC={max(lr_auc, rf_auc):.3f})')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
fig.savefig(out_path / f'figures/roc_curve_{best_name}.png')
plt.close(fig)


# Feature importance for RandomForest (if rf is best)
if best_name == 'random_forest':
# Need to extract feature names after one-hot encoding
ohe = best_pipe.named_steps['preproc'].transformers_[1][1].named_steps['onehot']
ohe_cols = ohe.get_feature_names_out(best_pipe.named_steps['preproc'].transformers_[1][2])
num_cols = best_pipe.named_steps['preproc'].transformers_[0][2]
feature_names = list(num_cols) + list(ohe_cols)
importances = best_pipe.named_steps['clf'].feature_importances_
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(20)
fig, ax = plt.subplots(figsize=(8,6))
feat_imp.plot(kind='barh', ax=ax)
ax.set_title('Top 20 feature importances (RandomForest)')
fig.savefig(out_path / 'figures/feature_importances_rf.png')
plt.close(fig)



# Save best model
joblib.dump(best_pipe, out_path / f'models/{best_name}.joblib')
print(f"Saved best model: {best_name}")




def main():
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='data/HR_comma_sep.csv', help='path to HR CSV')
parser.add_argument('--out', type=str, default='outputs', help='output folder')
args = parser.parse_args()


data_path = Path(args.data)
out_path = Path(args.out)
ensure_dirs(out_path)


if not data_path.exists():
# fallback to /mnt/data if running in environments like this chat
fallback = Path('/mnt/data/HR_comma_sep.csv')
if fallback.exists():
data_path = fallback
els
