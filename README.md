# Food Delivery Time Prediction using Machine Learning

## Overview

This project predicts the estimated food delivery time based on delivery partner information, restaurant and customer locations, weather conditions, traffic density, vehicle details, and order characteristics.

The objective is to build an accurate machine learning model that can help logistics companies estimate delivery times, improve customer satisfaction, and optimize delivery operations.

---

## Project Objectives

- Perform data inspection and preprocessing
- Handle missing values and inconsistent data
- Engineer meaningful features from raw data
- Compare multiple machine learning algorithms
- Predict delivery time accurately
- Visualize insights using Tableau
- Deploy a prediction script for new delivery requests

---

## Dataset

The dataset contains delivery information such as:

- Delivery Partner Details
- Restaurant Location
- Customer Location
- Weather Conditions
- Road Traffic Density
- Vehicle Condition
- Vehicle Type
- Order Type
- Festival Information
- City
- Delivery Time

Target Variable:

```
Time_taken(min)
```

---

## Project Structure

```
Food_Delivery_Time_Prediction/
│
├── data/
│   ├── updated.csv
│   ├── cleaned_test.csv
│   ├── encoded_cleaned_test.csv
│
├── notebooks/
│   ├── 01_Data_Inspection.ipynb
│   ├── 02_Data_Cleaning.ipynb
│   ├── 03_EDA.ipynb
│   ├── 04_Feature_Engineering.ipynb
│   ├── 05_Modeling.ipynb
│   ├── food_delivery_model.pkl
│   └── predict_delivery_time.py
│
├── dashboard/
│   └── Food_Delivery_Dashboard.twb
│
├── requirements.txt
├── README.md
```

---

## Data Preprocessing

The following preprocessing steps were performed:

- Removed unnecessary columns
- Removed duplicate records
- Handled missing values
- Corrected invalid time values
- Created preparation time feature
- Created delivery distance using Haversine Formula
- Generated peak hour indicator
- Created driver rating categories
- Created driver age groups
- Created distance categories

---

## Feature Engineering

New features include:

- Delivery_Distance_km
- Preparation_Time
- Order_Hour
- Pickup_Hour
- Peak_Hour
- Driver_Category
- Age_Group
- Distance_Category

---

## Machine Learning Models

The following models were trained and compared:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

Random Forest produced the best performance and was selected as the final model.

---

## Model Evaluation Metrics

The models were evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## Feature Importance

Random Forest feature importance analysis identified the most influential variables affecting delivery time.

Examples include:

- Delivery Distance
- Preparation Time
- Road Traffic Density
- Driver Rating
- Weather Conditions

---

## Tableau Dashboard

The project includes an interactive Tableau dashboard showing:

- Delivery Time Distribution
- Traffic Analysis
- Weather Impact
- Driver Performance
- Feature Importance
- Geographic Insights

---

## Prediction Script

The project includes a standalone prediction script.

```
predict_delivery_time.py
```

The script:

- Loads the trained Random Forest model
- Accepts new delivery information
- Performs feature engineering
- Predicts delivery time

Example:

```
Predicted Delivery Time : 27.5 minutes
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Joblib
- Tableau

---

## Future Improvements

- Hyperparameter Optimization
- XGBoost Implementation
- LightGBM Model
- Flask/FastAPI Deployment
- Real-time Prediction API
- Live GPS Integration
- Traffic API Integration

---

## Author

Saurabh Chaudhary
