import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

num_houses = 100
data = {
    'square_footage': np.random.randint(800, 3500, num_houses),
    'bedrooms': np.random.randint(1, 6, num_houses),
    'bathrooms': np.random.randint(1, 4, num_houses),
    'age': np.random.randint(0, 50, num_houses),
    'neighborhood': np.random.choice(['A', 'B', 'C'], num_houses),
    'price': np.random.randint(100000, 500000, num_houses)  # We'll modify this to make realistic
}

df = pd.DataFrame(data)

df['square_footage'] = df['square_footage'].fillna(df['square_footage'].mean()) 
df['bedrooms'] = df['bedrooms'].fillna(df['bedrooms'].mean()) 
df['bathrooms'] = df['bathrooms'].fillna(df['bathrooms'].mean()) 
df['age'] = df['age'].fillna(df['age'].mean()) 
df['price'] = df['price'].fillna(df['price'].mean()) 

le = LabelEncoder()
df['neighborhood'] = le.fit_transform(df['neighborhood'])
df['neighborhood'] = df['neighborhood'].fillna(df['neighborhood'].mean()) 

x = df[['square_footage','bedrooms','bathrooms','age','neighborhood']]
y = df['price']

model = LinearRegression()

X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model.fit(X_train,Y_train)

PredictionLR = model.predict(X_test)


print(f'Prdictions: {PredictionLR}')

accuracy = r2_score(Y_test,PredictionLR)
print(f'Testing Accuracy: {accuracy:.4f}')

mse = mean_squared_error(Y_test,PredictionLR)
print(f"MSE: {mse:.4f}")

print(f'For new Dataset......')

new_house = {
    'square_footage': [1800],
    'bedrooms': [3],
    'bathrooms': [2],
    'age': [10],
    'neighborhood': ['B']  # Assuming this was one of the neighborhoods in the data
}

df1 = pd.DataFrame(new_house)

df1['neighborhood'] = le.fit_transform(df1['neighborhood'])

x1 = df1[['square_footage','bedrooms','bathrooms','age','neighborhood']]

PredictionLR1 = model.predict(x1)

print(f'Prdictions: {PredictionLR1}')
