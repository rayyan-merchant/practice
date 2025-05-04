from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling (important for distance-based algorithms)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create KNN classifier
knn = KNeighborsClassifier(
    n_neighbors=5,  # Number of neighbors to use
    weights='uniform',  # 'uniform' or 'distance' (weight by inverse distance)
    algorithm='auto',  # Algorithm to compute neighbors: 'auto', 'ball_tree', 'kd_tree', 'brute'
    p=2  # Power parameter for Minkowski metric (1=Manhattan, 2=Euclidean)
)

# Train the model
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# To predict probabilities (for probability-based evaluation)
probabilities = knn.predict_proba(X_test)
print("\nPrediction probabilities for first 5 samples:")
print(probabilities[:5])
