# Support Vector Machine (SVM) Classification Template

This template demonstrates how to apply a Support Vector Machine (SVM) classifier to a binary classification problem using Python and scikit-learn. The code includes data preprocessing, model training, evaluation, and hyperparameter tuning.

```python
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------
# 1. Load and Prepare the Dataset
# ------------------------------

# Load a sample dataset (e.g., Iris dataset)
# Replace this with your own dataset (e.g., CSV file: pd.read_csv('your_data.csv'))
data = datasets.load_iris()
X = data.data  # Features
y = data.target  # Target labels

# Convert to binary classification (e.g., class 0 vs. others)
# Modify this based on your problem (e.g., select specific classes or use original labels)
# Examples of binary classification conversion:
# 1. One class vs. others (e.g., Setosa vs. non-Setosa):
#    y = (y == 0).astype(int)  # 1 for class 0 (Setosa), 0 for others
# 2. Two specific classes (e.g., Versicolor vs. Virginica, ignoring Setosa):
#    mask = (y == 1) | (y == 2)  # Select classes 1 and 2
#    X = X[mask]
#    y = y[mask]
#    y = (y == 1).astype(int)  # 1 for Versicolor, 0 for Virginica
# 3. Combining classes (e.g., Medium/High vs. Low severity):
#    y = (y >= 1).astype(int)  # 1 for Medium or High, 0 for Low
# 4. Custom labels (e.g., positive vs. non-positive in sentiment analysis):
#    y = (y == 'positive').astype(int)  # Assuming string labels
# Choose the appropriate transformation based on your problem
y = (y == 0).astype(int)  # Example: Setosa vs. non-Setosa

# Split the dataset into training and testing sets
# test_size=0.3 means 30% for testing, 70% for training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ------------------------------
# 2. Preprocess the Data
# ------------------------------

# Scale the features (important for SVM as it relies on distances)
# StandardScaler standardizes features to have mean=0 and variance=1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------
# 3. Train the SVM Model
# ------------------------------

# Initialize the SVM classifier
# kernel='rbf' for non-linear boundaries; use 'linear' for linearly separable data
# C controls the trade-off between margin maximization and classification error
# gamma='scale' automatically sets the kernel scale based on data
svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

# Train the model on the scaled training data
svm.fit(X_train_scaled, y_train)

# ------------------------------
# 4. Make Predictions
# ------------------------------

# Predict labels for the test set
y_pred = svm.predict(X_test_scaled)

# ------------------------------
# 5. Evaluate the Model
# ------------------------------

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Print detailed classification report (precision, recall, F1-score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ------------------------------
# 6. Hyperparameter Tuning (Optional)
# ------------------------------

# Define parameter grid for GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'gamma': ['scale', 'auto', 0.1, 0.01],  # Kernel coefficient
    'kernel': ['rbf', 'linear']  # Kernel types
}

# Initialize GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(
    SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1
)

# Fit GridSearchCV on the scaled training data
grid_search.fit(X_train_scaled, y_train)

# Print the best parameters and score
print("\nBest Parameters from GridSearchCV:")
print(grid_search.best_params_)
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")

# Use the best model to make predictions
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test_scaled)

# Evaluate the best model
best_accuracy = accuracy_score(y_test, y_pred_best)
print(f"\nAccuracy with Best Model: {best_accuracy:.4f}")
```

## Explanation of Key Steps
1. **Loading Data**: The Iris dataset is used as an example. Replace it with your dataset (e.g., a CSV file) and ensure the target variable is suitable for classification.
2. **Converting to Binary Classification**:
   - The target variable `y` is transformed to represent two classes.
   - Use conditional statements (e.g., `y == 0`) to assign binary labels (0 or 1).
   - For selecting specific classes, filter `X` and `y` using a mask to include only the desired classes.
   - Ensure the transformation aligns with the problem context (e.g., one class vs. others, two classes, or combining classes).
   - Check for class imbalance after conversion and consider techniques like `class_weight='balanced'` if needed.
3. **Preprocessing**: Features are scaled using `StandardScaler` because SVM is sensitive to the scale of input features. This step ensures all features contribute equally to the distance calculations.
4. **Training**: The SVM model is initialized with an RBF kernel for non-linear problems. The `C` parameter controls the penalty for misclassifications, and `gamma` influences the shape of the decision boundary.
5. **Evaluation**: Accuracy, classification report, and confusion matrix provide a comprehensive view of model performance.
6. **Hyperparameter Tuning**: `GridSearchCV` is used to find the optimal `C`, `gamma`, and `kernel` values, improving model performance.

## When to Use This Template
- Use this template for binary or multi-class classification problems.
- Modify the dataset loading step to work with your data (e.g., CSV, database).
- Adjust the binary classification conversion step based on your problem’s requirements (e.g., one class vs. others, two classes, or combining classes).
- Experiment with different kernels (`linear`, `rbf`, `poly`) based on the data’s separability.
- Use hyperparameter tuning for better performance, especially with complex datasets.

## Notes
- SVM is computationally expensive for very large datasets. Consider alternatives like Random Forest or Gradient Boosting for big data.
- Ensure the dataset is clean (no missing values) before applying SVM.
- If the classes are imbalanced, consider using `class_weight='balanced'` in `SVC` or oversampling techniques.
