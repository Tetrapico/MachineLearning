# Import necessary libraries
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
data = load_iris()
X = data.data  # Features
y = data.target  # Labels

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the KNN classifier (with K=3)
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Function to take user input for prediction
def user_input():
    print("Enter the features of a new sample (e.g., Sepal Length, Sepal Width, Petal Length, Petal Width):")
    try:
        # Get user input for each feature
        sepal_length = float(input("Sepal Length: "))
        sepal_width = float(input("Sepal Width: "))
        petal_length = float(input("Petal Length: "))
        petal_width = float(input("Petal Width: "))
        
        # Create a 2D array from the input to match the model's expected input shape
        new_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Predict the class for the new data point
        prediction = knn.predict(new_data)
        
        # Output the prediction
        print(f"The predicted class is: {data.target_names[prediction][0]}")
    except ValueError:
        print("Please enter valid numbers.")

# Get user input and predict
user_input()
