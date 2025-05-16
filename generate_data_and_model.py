import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

np.random.seed(42)

# Generate sample data
data = {
    'Age': np.random.randint(18, 80, size=1000),
    'SMS_Received': np.random.choice([0, 1], size=1000, p=[0.4, 0.6]),
    'Weekday': np.random.choice(range(1,8), size=1000),
    'Previous_NoShows': np.random.randint(0, 5, size=1000),
    'Diabetes': np.random.choice([0, 1], size=1000, p=[0.7, 0.3]),
    'Hypertension': np.random.choice([0, 1], size=1000, p=[0.6, 0.4]),
    'Medical Insurance': np.random.choice([0, 1], size=1000, p=[0.8, 0.2]),
    'NoShow': np.random.choice([0, 1], size=1000, p=[0.75, 0.25])
}

df = pd.DataFrame(data)
df.to_csv('appointments.csv', index=False)

# Train model
X = df[['Age', 'SMS_Received', 'Weekday', 'Previous_NoShows', 'Diabetes', 'Hypertension', 'Medical Insurance']]
y = df['NoShow']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'no_show_model.pkl')

print("Data and model generated successfully!")
