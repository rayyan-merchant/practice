# Linear Regression Template
# ---------------------------------
# This template covers:
# 1. Reading a CSV dataset
# 2. Handling missing values (dropna, manual imputation, SimpleImputer)
# 3. Encoding categorical features (Label Encoding, One-Hot Encoding)
# 4. Exploratory Data Analysis (EDA)
# 5. Train-test split and optional scaling
# 6. K-Fold Cross Validation
# 7. Leave-One-Out Cross Validation (LOOCV)
# 8. Model evaluation and performance metrics (MSE, RMSE, MAE, R2)
# 9. Visualization of performance metrics and residuals

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -----------------------------
# 1. Load the dataset
# -----------------------------
df = pd.read_csv('your_dataset.csv')  # <-- Replace with your file path

# -----------------------------
# 2. Handling Missing Values
# -----------------------------
# Option A: Drop rows with any missing values
df_dropna = df.dropna()

# Option B: Manual imputation for numeric and categorical
# Numeric: median; Categorical: mode
df_manual = df.copy()
for col in df_manual.select_dtypes(include=[np.number]).columns:
    df_manual[col].fillna(df_manual[col].median(), inplace=True)
for col in df_manual.select_dtypes(include=['object', 'category']).columns:
    df_manual[col].fillna(df_manual[col].mode()[0], inplace=True)

# Option C: Automated imputation using SimpleImputer
num_cols = df.select_dtypes(include=[np.number]).columns\cat_cols = df.select_dtypes(include=['object', 'category']).columns
num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')

df_imputed = df.copy()
df_imputed[num_cols] = num_imputer.fit_transform(df_imputed[num_cols])
df_imputed[cat_cols] = cat_imputer.fit_transform(df_imputed[cat_cols])

# Choose DataFrame to proceed with
df = df_imputed.copy()

# -----------------------------
# 3. Encoding Categorical Variables
# -----------------------------
# Label encode binary features if any
label_enc = LabelEncoder()
# Example: encode a binary column 'binary_col' if present
# df['binary_col'] = label_enc.fit_transform(df['binary_col'])

# One-Hot encode other categorical features
df = pd.get_dummies(df, drop_first=True)

# -----------------------------
# 4. Exploratory Data Analysis (EDA)
# -----------------------------
print(df.head())
print(df.info())
print(df.describe())

# Pairplot to see relationships (sample if too large)
sns.pairplot(df.sample(min(1000, len(df))), diag_kind='kde')
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()

# -----------------------------
# 5. Train-Test Split and Scaling
# -----------------------------
# Define features X and target y
target_col = 'target'  # <-- Replace with your target column name
X = df.drop(target_col, axis=1)
y = df[target_col]

# Split into train and test sets
test_size = 0.2  # 20% for test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

# Optional: Feature scaling (often improves convergence)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 6. K-Fold Cross Validation
# -----------------------------
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
model_lr = LinearRegression()

# Use negative MSE for scoring to maximize
cv_mse = cross_val_score(model_lr, X_train_scaled, y_train, cv=kf, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_mse)
print(f"K-Fold CV ({k} folds) RMSE: {cv_rmse}")
print(f"Mean CV RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")

# -----------------------------
# 7. Leave-One-Out Cross Validation (LOOCV)
# -----------------------------
loo = LeaveOneOut()
loo_mse = cross_val_score(model_lr, X_train_scaled, y_train, cv=loo, scoring='neg_mean_squared_error')
loo_rmse = np.sqrt(-loo_mse)
print(f"LOOCV RMSE: {loo_rmse.mean():.4f}")

# -----------------------------
# 8. Model Training and Evaluation
# -----------------------------
# Fit model on entire training set
model_lr.fit(X_train_scaled, y_train)
# Predictions on test set
y_pred = model_lr.predict(X_test_scaled)

# Compute performance metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Test Set Performance:")
print(f"MSE : {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R2  : {r2:.4f}")

# -----------------------------
# 9. Visualization of Results
# -----------------------------
# 9.1. Predicted vs Actual
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Values')
plt.show()

# 9.2. Residuals plot
residuals = y_test - y_pred
plt.figure(figsize=(6, 4))
sns.histplot(residuals, kde=True)
plt.title('Residuals Distribution')
plt.xlabel('Residual')
plt.show()

# 9.3. Residuals vs Fitted
plt.figure(figsize=(6, 4))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')
plt.show()
