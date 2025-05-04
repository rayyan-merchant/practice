# Decision Tree Classification Template

This template demonstrates how to apply a Decision Tree Classifier to a classification problem using Python and scikit-learn. The code includes data loading, preprocessing, model training, prediction, evaluation, and visualization of the decision tree. It can be adapted for regression by using `DecisionTreeRegressor`.

## Prerequisites
- Install required libraries: `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `graphviz` (for tree visualization)
- Run the following command to install dependencies:
  ```bash
  pip install scikit-learn numpy pandas matplotlib graphviz
  ```
- Install Graphviz executable for tree visualization (e.g., via `apt-get install graphviz` on Linux or downloading from the Graphviz website).

## Python Code

```python
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import graphviz
from sklearn import datasets

# ------------------------------
# 1. Load and Prepare the Dataset
# ------------------------------

# Load a sample dataset (e.g., Iris dataset)
# Replace this with your own dataset (e.g., CSV file: pd.read_csv('your_data.csv'))
data = datasets.load_iris()
X = data.data  # Features
y = data.target  # Target labels

# Convert to binary classification (optional, e.g., class 0 vs. others)
# Example: Classify Setosa (class 0) vs. non-Setosa
y = (y == 0).astype(int)  # 1 for Setosa, 0 for others
# For multi-class or original labels, skip the above step
# Ensure X is a 2D array (shape: n_samples, n_features) and y is a 1D array

# Split the dataset into training and testing sets
# test_size=0.2 means 20% for testing, 80% for training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------
# 2. Preprocess the Data
# ------------------------------

# Scale the features (optional for decision trees)
# Decision trees are not sensitive to feature scales, but scaling can be applied for consistency
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------
# 3. Train the Decision Tree Model
# ------------------------------

# Initialize the DecisionTreeClassifier
# Parameters:
# - max_depth: Limit tree depth to prevent overfitting
# - min_samples_split: Minimum samples required to split a node
# - min_samples_leaf: Minimum samples required at a leaf node
# - random_state: Ensure reproducibility
DT = DecisionTreeClassifier(max_depth=3, min_samples_split=2, min_samples_leaf=1, random_state=42)

# Train the model on the training data
# Use scaled or unscaled data (decision trees are invariant to scaling)
ModelDT = DT.fit(X_train_scaled, y_train)

# ------------------------------
# 4. Make Predictions
# ------------------------------

# Predict on the test data
PredictionDT = ModelDT.predict(X_test_scaled)

# Print the predictions
print("\nPredictions:", PredictionDT)

# ------------------------------
# 5. Evaluate the Model
# ------------------------------

# Calculate training accuracy
# The score method returns the mean accuracy on the given data
tracDT = ModelDT.score(X_train_scaled, y_train)
TrainingAccDT = tracDT * 100
print("\n==================== DT Training Accuracy ===================")
print(f"Training Accuracy: {TrainingAccDT:.2f}%")

# Calculate testing accuracy
teacDT = accuracy_score(y_test, PredictionDT)
testingAccDT = teacDT * 100
print("\n==================== DT Testing Accuracy ===================")
print(f"Testing Accuracy: {testingAccDT:.2f}%")

# Print detailed classification report (precision, recall, F1-score)
print("\nClassification Report:")
print(classification_report(y_test, PredictionDT))

# Print confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, PredictionDT))

# ------------------------------
# 6. Visualize the Decision Tree (Optional)
# ------------------------------

# Export the decision tree to a DOT file
dot_data = export_graphviz(
    ModelDT,
    out_file=None,
    feature_names=data.feature_names,
    class_names=['Non-Setosa', 'Setosa'],  # Adjust based on your classes
    filled=True,
    rounded=True,
    special_characters=True
)

# Render the tree using graphviz
graph = graphviz.Source(dot_data)
graph.render("decision_tree", format="png", cleanup=True)  # Saves as decision_tree.png
print("\nDecision tree visualization saved as 'decision_tree.png'")

# ------------------------------
# 7. Feature Importance (Optional)
# ------------------------------

# Calculate and print feature importance
# Feature importance indicates the contribution of each feature to the splits
feature_importance = ModelDT.feature_importances_
feature_names = data.feature_names  # Adjust for your dataset
print("\nFeature Importance:")
for name, importance in zip(feature_names, feature_importance):
    print(f"{name}: {importance:.4f}")

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.bar(feature_names, feature_importance)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance in Decision Tree')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()
```

## Explanation of Key Steps
1. **Loading Data**:
   - The Iris dataset is used as an example. Replace it with your dataset (e.g., CSV file).
   - The target variable `y` can be binary or multi-class. For binary classification, transform `y` as needed (e.g., one class vs. others).
   - Ensure `X` is a 2D array and `y` is a 1D array of labels.
2. **Preprocessing**:
   - Split the data into training and testing sets to evaluate model performance.
   - Scaling is optional for decision trees, as they are not sensitive to feature scales. However, it’s included for consistency with other models.
3. **Training**:
   - The `DecisionTreeClassifier` builds a tree by recursively splitting the data based on the best attribute (using metrics like Gini impurity or entropy).
   - Parameters like `max_depth`, `min_samples_split`, and `min_samples_leaf` control tree complexity to prevent overfitting.
4. **Prediction**:
   - Use the trained model to predict class labels for the test set.
5. **Evaluation**:
   - **Training Accuracy**: Measures how well the model fits the training data (beware of overfitting if too high).
   - **Testing Accuracy**: Measures generalization to unseen data.
   - **Classification Report**: Provides precision, recall, and F1-score for each class.
   - **Confusion Matrix**: Shows true positives, false positives, true negatives, and false negatives.
6. **Visualization**:
   - The decision tree is visualized as a flowchart using `graphviz`, showing splits, nodes, and class labels.
   - Saved as a PNG file for easy inspection.
7. **Feature Importance**:
   - Decision trees provide feature importance scores based on how much each feature contributes to reducing impurity.
   - A bar plot visualizes the importance of each feature.

## When to Use This Template
- Use this template for classification problems (binary or multi-class) or regression problems (by replacing `DecisionTreeClassifier` with `DecisionTreeRegressor`).
- Modify the dataset loading step to work with your data (e.g., CSV, database).
- Adjust the binary classification conversion step if needed (e.g., one class vs. others, two classes).
- Use visualization and feature importance to interpret the model and identify key features.
- Tune hyperparameters (`max_depth`, `min_samples_split`, etc.) to balance bias and variance.

## Notes
- **Overfitting**: Decision trees are prone to overfitting, especially with deep trees. Use pruning parameters (`max_depth`, `min_samples_leaf`) or ensemble methods like Random Forest.
- **Categorical Features**: Scikit-learn’s decision trees require numerical inputs. Encode categorical features using one-hot encoding or label encoding before training.
- **Imbalanced Data**: If classes are imbalanced, use `class_weight='balanced'` in `DecisionTreeClassifier` or oversampling techniques.
- **Visualization**: The `graphviz` library requires the Graphviz executable to be installed. Ensure it’s set up for tree visualization.
- **Regression Adaptation**:
   - For regression, use `DecisionTreeRegressor` and replace accuracy metrics with regression metrics like MSE, RMSE, and R-squared (see the Linear Regression template).
   - Example: `from sklearn.tree import DecisionTreeRegressor`

## Example Applications
- **Classification**: Predicting whether a customer will buy a product based on age, income, and browsing history.
- **Regression**: Predicting house prices based on size, location, and number of bedrooms.
- **Multi-Class**: Classifying types of flowers (e.g., Iris dataset) based on petal and sepal measurements.