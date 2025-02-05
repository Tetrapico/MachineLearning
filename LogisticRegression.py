import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Sample data (Age, Glucose Level, Blood Pressure)
data = np.array([
    [50, 100, 80],
    [30, 120, 85],
    [45, 160, 90],
    [60, 130, 75],
    [25, 110, 85],
    [70, 170, 90],
    [35, 105, 80]
])

# Labels (1 = Diabetes, 0 = No Diabetes)
labels = np.array([1, 0, 1, 0, 0, 1, 0])

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)
age = float(input("Enter age: "))
glucose = float(input("Enter glucose level: "))
blood_pressure = float(input("Enter blood pressure: "))

# Reshaping the input to match the model's expected input
user_input = np.array([[age, glucose, blood_pressure]])

# Making the prediction
prediction = model.predict(user_input)

# Output the result
if prediction[0] == 1:
    print("Prediction: Diabetes")
else:
    print("Prediction: No Diabetes")
