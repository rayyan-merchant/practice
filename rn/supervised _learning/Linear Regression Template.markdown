# Linear Regression Template

This template demonstrates how to apply a Linear Regression model to a regression problem using Python and scikit-learn. The code includes data loading, preprocessing, model training, prediction, and evaluation with metrics like R-squared and Root Mean Squared Error (RMSE).

## Prerequisites
- Install required libraries: `scikit-learn`, `numpy`, `pandas`, `matplotlib` (for visualization)
- Run the following command to install dependencies:
  ```bash
  pip install scikit-learn numpy pandas matplotlib
  ```

## Python Code

```python
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ------------------------------
# 1. Load and Prepare the Dataset
# ------------------------------

# Load a sample dataset (e.g., synthetic data or your own dataset)
# Replace this with your dataset (e.g., CSV file: pd.read_csv('your_data.csv'))
# Example: Synthetic dataset with one feature
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Single feature (independent variable)
y = 3 * X.squeeze() + 5 + np.random.randn(100) * 2  # Linear relationship with noise

# If using multiple features, X should be a 2D array (e.g., X = data[['feature1', 'feature2']])
# Ensure X is a 2D array even for simple linear regression (shape: n_samples, n_features)
X = X.reshape(-1, 1)  # Reshape for scikit-learn compatibility

# ------------------------------
# 2. Preprocess the Data
# ------------------------------

# Split the data into training and testing sets
# test_size=0.2 means 20% for testing, 80% for training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the features (optional, but recommended for multiple features)
# StandardScaler standardizes features to have mean=0 and variance=1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------
# 3. Train the Linear Regression Model
# ------------------------------

# Create and train the Linear Regression model
# LinearRegression fits a line: y = B0 + B1*X (simple) or y = B0 + B1*X1 + ... + Bn*Xn (multiple)
LR = LinearRegression()
ModelLR = LR.fit(X_train_scaled, y_train)

# Print the model parameters
# Intercept (B0) and slope(s) (B1, B2, ...) for the best-fit line
print(f"Intercept (B0): {ModelLR.intercept_:.4f}")
print(f"Slope(s) (B1, ...): {ModelLR.coef_}")

# ------------------------------
# 4. Make Predictions
# ------------------------------

# Predict on the test data
PredictionLR = ModelLR.predict(X_test_scaled)

# Print the predictions
print("\nPredictions:", PredictionLR)

# ------------------------------
# 5. Evaluate the Model
# ------------------------------

# Calculate Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
mse = mean_squared_error(y_test, PredictionLR)
rmse = np.sqrt(mse)

# Calculate R-squared (coefficient of determination)
r2 = r2_score(y_test, PredictionLR)

# Print evaluation metrics
print("\n=================== Linear Regression Evaluation ===================")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R2): {r2:.4f}")
print(f"Testing Accuracy (R2 * 100): {r2 * 100:.2f}%")

# ------------------------------
# 6. Visualize the Results (Optional)
# ------------------------------

# Plot the data and best-fit line (for simple linear regression with one feature)
if X.shape[1] == 1:  # Only plot for simple linear regression
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, color='blue', label='Training data')
    plt.scatter(X_test, y_test, color='green', label='Test data')
    # Plot the best-fit line using the full range of X
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    X_range_scaled = scaler.transform(X_range)
    y_pred_range = ModelLR.predict(X_range_scaled)
    plt.plot(X_range, y_pred_range, color='red', label='Best-fit line')
    plt.xlabel('Independent Variable (X)')
    plt.ylabel('Dependent Variable (y)')
    plt.title('Linear Regression: Best-Fit Line')
    plt.legend()
    plt.grid(True)
    plt.savefig('linear_regression_plot.png')
    plt.close()

# ------------------------------
# 7. Residual Analysis (Optional)
# ------------------------------

# Calculate residuals (errors): y_predicted - y_actual
residuals = PredictionLR - y_test

# Plot residuals vs. predicted values to check for homoscedasticity
plt.figure(figsize=(10, 6))
plt.scatter(PredictionLR, residuals, color='purple')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True)
plt.savefig('residual_plot.png')
plt.close()
```

## Explanation of Key Steps
1. **Loading Data**:
   - A synthetic dataset is used for demonstration. Replace it with your dataset (e.g., CSV file or other source).
   - Ensure `X` is a 2D array (even for simple linear regression) and `y` is a 1D array of continuous values.
2. **Preprocessing**:
   - Split the data into training and testing sets to evaluate model performance.
   - Scale features using `StandardScaler` (optional for simple linear regression but recommended for multiple features to ensure consistent scales).
3. **Training**:
   - The `LinearRegression` model fits a line of the form `y = B0 + B1*X` (simple) or `y = B0 + B1*X1 + ... + Bn*Xn` (multiple).
   - The model optimizes coefficients (B0, B1, ...) to minimize the sum of squared residuals.
4. **Prediction**:
   - Use the trained model to predict `y` values for the test set.
5. **Evaluation**:
   - **Mean Squared Error (MSE)**: Average of squared differences between predicted and actual values.
   - **Root Mean Squared Error (RMSE)**: Square root of MSE, representing the average prediction error in the same units as `y`.
   - **R-squared (R2)**: Proportion of variance in `y` explained by the model (ranges from 0 to 1; higher is better).
6. **Visualization**:
   - For simple linear regression, plot the data points and best-fit line to visualize the linear relationship.
   - Plot residuals vs. predicted values to check for homoscedasticity (constant variance of errors).
7. **Residual Analysis**:
   - Residuals are calculated as `y_predicted - y_actual`.
   - A residual plot helps verify assumptions like homoscedasticity and can reveal patterns indicating non-linearity.

## When to Use This Template
- Use this template for regression problems with a continuous dependent variable.
- Modify the dataset loading step to work with your data (e.g., CSV, database).
- For simple linear regression, ensure `X` has one feature; for multiple linear regression, include multiple features in `X`.
- Use visualization and residual analysis to validate the linear assumption and model fit.
- If the relationship is non-linear, consider polynomial features or other models (e.g., decision trees, random forests).

## Notes
- **Assumption Checks**:
  - Verify linearity using scatter plots or correlation analysis.
  - Check for homoscedasticity and normality of residuals using residual plots and histograms.
  - Detect multicollinearity in multiple linear regression using correlation matrices or VIF.
- **Preprocessing**:
  - Handle missing values and outliers before training.
  - Scale features if they have different units or ranges.
- **Limitations**:
  - Linear regression assumes a linear relationship. For non-linear data, consider polynomial regression or non-linear models.
  - Sensitive to outliers, which can skew the best-fit line.
- **Extensions**:
  - Use `PolynomialFeatures` from scikit-learn for non-linear relationships.
  - Apply regularization (e.g., Ridge or Lasso regression) for high-dimensional data or multicollinearity.
