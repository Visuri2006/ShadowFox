import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# =========================
# LOAD DATASET
# =========================

print("Program Started Successfully!")

df = pd.read_csv("dataset/car.csv")

print("\nDataset Loaded Successfully!")

# =========================
# DISPLAY DATASET
# =========================

print("\nFirst 5 Rows:\n")
print(df.head())

# =========================
# CHECK DATASET INFO
# =========================

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# =========================
# DATA PREPROCESSING
# =========================

# Convert categorical columns into numerical values

df.replace({
    'Fuel_Type': {
        'Petrol': 0,
        'Diesel': 1,
        'CNG': 2
    },
    'Seller_Type': {
        'Dealer': 0,
        'Individual': 1
    },
    'Transmission': {
        'Manual': 0,
        'Automatic': 1
    }
}, inplace=True)

print("\nData After Encoding:\n")
print(df.head())

# =========================
# SPLITTING FEATURES & TARGET
# =========================

X = df.drop(['Car_Name', 'Selling_Price'], axis=1)

Y = df['Selling_Price']

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.1,
    random_state=2
)

# =========================
# MODEL TRAINING
# =========================

model = RandomForestRegressor()

model.fit(X_train, Y_train)

print("\nModel Trained Successfully!")

# =========================
# MODEL PREDICTION
# =========================

training_data_prediction = model.predict(X_train)

# =========================
# ACCURACY CHECK
# =========================

error_score = r2_score(Y_train, training_data_prediction)

print("\nR2 Score:", error_score)

# =========================
# TEST PREDICTION
# =========================

test_data_prediction = model.predict(X_test)

print("\nSample Predictions:\n")

for i in range(5):
    print("Actual Price:", Y_test.iloc[i],
          "| Predicted Price:", round(test_data_prediction[i], 2))

print("\nCar Price Prediction System Completed Successfully!")
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# SELLING PRICE DISTRIBUTION
# =========================

plt.figure(figsize=(8, 5))

sns.histplot(df['Selling_Price'], bins=30, kde=True)

plt.title("Selling Price Distribution")
plt.xlabel("Selling Price")
plt.ylabel("Frequency")

# Save graph
plt.savefig("images/selling_price_distribution.png")

plt.show()

# =========================
# CORRELATION HEATMAP
# =========================

plt.figure(figsize=(10, 6))

numeric_df = df.select_dtypes(include=np.number)

sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")

plt.title("Correlation Heatmap")

# Save graph
plt.savefig("images/correlation_heatmap.png")

plt.show()

# =========================
# FUEL TYPE COUNT
# =========================

plt.figure(figsize=(6, 5))

sns.countplot(x='Fuel_Type', data=df)

plt.title("Fuel Type Distribution")
plt.xlabel("Fuel Type")
plt.ylabel("Count")

# Save graph
plt.savefig("images/fuel_type_distribution.png")

plt.show()

# =========================
# TRANSMISSION TYPE COUNT
# =========================

plt.figure(figsize=(6, 5))

sns.countplot(x='Transmission', data=df)

plt.title("Transmission Type Distribution")
plt.xlabel("Transmission")
plt.ylabel("Count")

# Save graph
plt.savefig("images/transmission_distribution.png")

plt.show()

# =========================
# YEAR VS SELLING PRICE
# =========================

plt.figure(figsize=(10, 5))

sns.scatterplot(x='Year', y='Selling_Price', data=df)

plt.title("Year vs Selling Price")
plt.xlabel("Year")
plt.ylabel("Selling Price")

# Save graph
plt.savefig("images/year_vs_selling_price.png")

plt.show()

print("\nVisualizations Generated Successfully!")