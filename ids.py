from sklearn.ensemble import RandomForestClassifier

# Sample training data
X = [[1, 20], [2, 30], [3, 40], [10, 80]]
y = ["NORMAL", "NORMAL", "NORMAL", "ATTACK"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Predict
prediction = model.predict([[8, 70]])

print("Prediction:", prediction[0])