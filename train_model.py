import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Load the dataset
data = pd.read_csv("credit_card_dataset.csv")

# Select features (input) and target (output)
X = data[["Age", "Income", "CreditScore"]]
y = data["Approved"]

# Create and train the model
model = LogisticRegression()
model.fit(X, y)

# Save the trained model
joblib.dump(model, "model.pkl")

print("=================================")
print("Model trained successfully!")
print("model.pkl has been created.")
print("=================================")