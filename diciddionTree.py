from sklearn.naive_bayes import GaussianNB
import numpy as np

# Encoding mappings for user input
weather_map = {"sunny": 0, "rainy": 1}
temp_map = {"hot": 0, "cold": 1}
windy_map = {"no": 0, "yes": 1}

# Training dataset (Weather, Temperature, Windy) and corresponding labels (0 = Stay Inside, 1 = Go Outside)
X = np.array([
    [0, 0, 0],  # Sunny, Hot, No
    [0, 1, 0],  # Sunny, Cold, No
    [1, 0, 1],  # Rainy, Hot, Yes
    [1, 1, 0],  # Rainy, Cold, No
    [1, 0, 0],  # Rainy, Hot, No
])

y = np.array([0, 1, 0, 1, 1])  # Labels (0 = Stay Inside, 1 = Go Outside)

# Train Naïve Bayes Model
model = GaussianNB()
model.fit(X, y)

# Taking user input
weather = input("Enter weather (sunny/rainy): ").strip().lower()
temperature = input("Enter temperature (hot/cold): ").strip().lower()
windy = input("Is it windy? (yes/no): ").strip().lower()

# Convert input to numerical values
if weather in weather_map and temperature in temp_map and windy in windy_map:
    user_data = np.array([[weather_map[weather], temp_map[temperature], windy_map[windy]]])
    
    # Prediction
    prediction = model.predict(user_data)
    
    print("\n🚀 Should you go outside? →", "YES 😃" if prediction[0] == 1 else "NO ☹️")
else:
    print("\n⚠️ Invalid input! Please enter values correctly.")
