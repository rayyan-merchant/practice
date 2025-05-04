import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_text

num_customers = 500

data = {
    'customer_id': range(1, num_customers+1),
    'age': np.random.randint(18, 70, size=num_customers),
    'total_spending': np.concatenate([
        np.random.normal(1200, 200, int(num_customers*0.3)),
        np.random.normal(400, 150, int(num_customers*0.7))
    ]),
    'num_visits': np.concatenate([
        np.random.poisson(12, int(num_customers*0.3)),
        np.random.poisson(4, int(num_customers*0.7))
    ]),
    'purchase_freq': np.concatenate([
        np.random.normal(15, 3, int(num_customers*0.3)),
        np.random.normal(45, 10, int(num_customers*0.7))
    ])
}

df = pd.DataFrame(data)

# Create target variable (high-value = 1, low-value = 0)
df['high_value'] = ((df['total_spending'] > 800) & (df['num_visits'] > 8) & (df['purchase_freq'] < 30)).astype(int)

for col in ['total_spending', 'num_visits', 'purchase_freq']:
    df[col].fillna(df[col].mean(),inplace=True)
    
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['age', 'total_spending', 'num_visits', 'purchase_freq']])

df_scaled = pd.DataFrame(scaled_features, columns=['age_scaled', 'total_spending_scaled', 'num_visits_scaled', 'purchase_freq_scaled'])
df = pd.concat([df, df_scaled], axis=1)

X = df[['age_scaled', 'total_spending_scaled', 'num_visits_scaled', 'purchase_freq_scaled']]
y = df['high_value']

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train SVM classifier
svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train, Y_train)

# Evaluate on test set
y_pred_svm = svm_classifier.predict(X_test)

print("\nSVM Classification Report:")
print(classification_report(Y_test, y_pred_svm))
print("\nSVM Confusion Matrix:")
print(confusion_matrix(Y_test, y_pred_svm))

# Train Decision Tree classifier
tree_classifier = DecisionTreeClassifier()
tree_classifier.fit(X_train, Y_train)

# Extract rules
tree_rules = export_text(tree_classifier, feature_names=['age', 'total_spending', 'num_visits', 'purchase_freq'])
print("\nDecision Tree Rules:")
print(tree_rules)

# Evaluate on test set
y_pred_tree = tree_classifier.predict(X_test)

print("\nDecision Tree Classification Report:")
print(classification_report(Y_test, y_pred_tree))
print("\nDecision Tree Confusion Matrix:")
print(confusion_matrix(Y_test, y_pred_tree))
