import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Prepare Data
X = np.array([600, 800, 1000, 1200, 1500, 1800, 2000]).reshape(-1, 1)  # Feature: Square Footage
y = np.array([150000, 180000, 210000, 250000, 290000, 320000, 350000])  # Target: House Price

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 3: Make Predictions
y_pred = model.predict(X_test)

# Step 4: Evaluate Model
# square_footage = float(input("Enter the square footage of the house: "))

# # Prepare the input data for prediction
# input_data = np.array([[square_footage]]) in this way we can take input form user 
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"Predicted prices: {y_pred}")
print(f"actual price of the text case :{y_test}")

# Step 5: Visualize
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, model.predict(X), color='red', linewidth=2, label="Regression Line")
plt.xlabel("Square Footage")
plt.ylabel("House Price")
plt.legend()
plt.show()
