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
    ((out_path / "figures").mkdir(parents=True, exist_ok=True)
     (out_path / "models").mkdir(parents=True, exist_ok=True))




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
els