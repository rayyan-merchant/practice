import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data = {
    'email_text': [
        "Congratulations! You've won a free vacation. Click here to claim your prize now!",
        "Hi John, just following up on our meeting tomorrow at 10am in conference room B.",
        "Limited time offer! Get 50% cash bonus if you respond within 30 minutes!",
        "Please review the attached project report and provide your feedback by EOD.",
        "URGENT: Your account has been selected for a special guaranteed bonus offer!",
        "Team, don't forget we have our weekly sync meeting today at 3pm.",
        "You're the winner of $10,000! Click the link to claim your free money now!!!",
        "Reminder: Your invoice #12345 is due next week. Please make the payment.",
        "Exclusive deal just for you! Buy one get one free - this offer won't last!",
        "Hi Mom, just checking in to see how you're doing. Call me when you get this."
    ],
    'email_length': [125, 98, 112, 87, 134, 76, 110, 92, 118, 85],
    'has_hyperlink': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    'sender_address': [
        "promo@winnergiveaways.com", 
        "jane.doe@company.com",
        "deals@exclusiveoffers.net",
        "manager@business.org",
        "alert@account-security.com",
        "team.lead@company.com",
        "notify@prizedraw.win",
        "billing@services.com",
        "special@deals4u.com",
        "family.personal@gmail.com"
    ],
    'is_spam': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

le = LabelEncoder()
df['email_text'] = le.fit_transform(df['email_text'])
df['sender_address'] = le.fit_transform(df['sender_address'])

model = SVC()

x = df[['email_text','email_length','has_hyperlink','sender_address']]
y = df['is_spam']

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=42)

svm = SVC(kernel='rbf', C=1, gamma='scale')
svm.fit(X_train,Y_train)

y_pred = svm.predict(X_test)

print(f'Prdictions: {y_pred}')

acc = accuracy_score(Y_test,y_pred)

print(acc)

new_email = {
    'email_text': ['Hello World'],
    'email_length': [100],
    'has_hyperlink': [0],
    'sender_address': ['raghib@gmail.com']
}

df1 = pd.DataFrame(new_email)
df1['email_text'] = le.fit_transform(df1['email_text'])
df1['sender_address'] = le.fit_transform(df1['sender_address'])

x1 = df1[['email_text','email_length','has_hyperlink','sender_address']]

y_pred1 = svm.predict(x1)

print(f'Prdictions: {y_pred1}')
