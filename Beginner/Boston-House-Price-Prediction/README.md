# Boston House Price Prediction

## Project Overview

This project predicts Boston house prices using Machine Learning.

The model is trained using housing features such as:

* Crime rate
* Number of rooms
* Property tax
* Distance to employment centers
* Other housing-related attributes

The goal is to predict house prices accurately using regression techniques.

---

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Matplotlib

---

## Machine Learning Workflow

### 1. Data Preprocessing

* Loaded dataset
* Handled missing values using mean imputation

### 2. Feature Selection

Selected all relevant housing features.

### 3. Model Training

Used Random Forest Regressor.

### 4. Model Evaluation

Performance Metrics:

* MAE: 2.0624
* MSE: 8.2275
* R² Score: 0.8878

---

## Visualization

Scatter plot comparing actual vs predicted house prices.

---

## Project Structure

Boston-House-Price-Prediction/
│── dataset/
│── src/
│── images/
│── README.md
│── requirements.txt

---

## Conclusion

The Random Forest model achieved strong prediction performance with an R² score of 0.8878, making it effective for house price prediction.
