import joblib
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# ==========================================
# Load Trained Model
# ==========================================
model = joblib.load(r"notebooks\food_delivery_model.pkl")

# ==========================================
# Function to Calculate Distance (Haversine)
# ==========================================

def haversine(lat1, lon1, lat2, lon2):

    R = 6371

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2

    c = 2 * atan2(sqrt(a), sqrt(1-a))

    return R * c


# ==========================================
# Enter New Order Details
# ==========================================

print("=" * 50)
print("FOOD DELIVERY TIME PREDICTION")
print("=" * 50)

new_order = {}

print("\nEnter Driver Details")
new_order["Delivery_person_Age"] = int(input("Driver Age: "))
new_order["Delivery_person_Ratings"] = float(input("Driver Rating (2.5 - 5): "))

print("\nEnter Restaurant Coordinates")
new_order["Restaurant_latitude"] = float(input("Restaurant Latitude: "))
new_order["Restaurant_longitude"] = float(input("Restaurant Longitude: "))

print("\nEnter Customer Coordinates")
new_order["Delivery_location_latitude"] = float(input("Customer Latitude: "))
new_order["Delivery_location_longitude"] = float(input("Customer Longitude: "))

print("\nEnter Encoded Values")

new_order["Weatherconditions"] = int(input("Weather (0-7): "))
new_order["Road_traffic_density"] = int(input("Traffic Density (0-5): "))
new_order["Vehicle_condition"] = int(input("Vehicle Condition (0-3): "))
new_order["Type_of_order"] = int(input("Order Type (0-3): "))
new_order["Type_of_vehicle"] = int(input("Vehicle Type (1-4): "))
new_order["multiple_deliveries"] = int(input("Multiple Deliveries (0-3): "))
new_order["Festival"] = int(input("Festival (1=No, 2=Yes): "))
new_order["City"] = int(input("City (Encoded): "))

print("\nTime Information")

new_order["Preparation_Time"] = float(input("Preparation Time (minutes): "))
new_order["Order_Hour"] = int(input("Order Hour (0-23): "))
new_order["Pickup_Hour"] = int(input("Pickup Hour (0-23): "))

# ==========================================
# Calculate Delivery Distance
# ==========================================

distance = haversine(
    new_order["Restaurant_latitude"],
    new_order["Restaurant_longitude"],
    new_order["Delivery_location_latitude"],
    new_order["Delivery_location_longitude"]
)

new_order["Delivery_Distance_km"] = round(distance, 2)

# ==========================================
# Peak Hour Feature
# ==========================================

peak_hours = [11,12,13,14,18,19,20,21]

new_order["Peak_Hour"] = int(new_order["Order_Hour"] in peak_hours)


# ==========================================
# Driver Category
# ==========================================

rating = new_order["Delivery_person_Ratings"]

new_order["Driver_Category_Excellent"] = int(rating >= 4.8)
new_order["Driver_Category_Good"] = int(4.5 <= rating < 4.8)

# ==========================================
# Age Group
# ==========================================

age = new_order["Delivery_person_Age"]


new_order["Age_Group_26-35"] = int(26 <= age <= 35)
new_order["Age_Group_36-45"] = int(36 <= age <= 45)


# ==========================================
# Distance Category
# ==========================================

d = new_order["Delivery_Distance_km"]

new_order["Distance_Category_Short"] = int(d < 5)
new_order["Distance_Category_Medium"] = int(5 <= d < 10)
new_order["Distance_Category_Very Long"] = int(d >= 15)

# ==========================================
# Create DataFrame
# ==========================================

features = pd.DataFrame([new_order])

# Keep only features used during training

features = features[[
'Delivery_person_Age',
'Delivery_person_Ratings',
'Restaurant_latitude',
'Restaurant_longitude',
'Delivery_location_latitude',
'Delivery_location_longitude',
'Weatherconditions',
'Road_traffic_density',
'Vehicle_condition',
'Type_of_order',
'Type_of_vehicle',
'multiple_deliveries',
'Festival',
'City',
'Delivery_Distance_km',
'Preparation_Time',
'Order_Hour',
'Pickup_Hour',
'Peak_Hour',
'Driver_Category_Excellent',
'Driver_Category_Good',
'Age_Group_26-35',
'Age_Group_36-45',
'Distance_Category_Medium',
'Distance_Category_Short',
'Distance_Category_Very Long'
]]

# ==========================================
# Prediction
# ==========================================

prediction = model.predict(features)

print("\n")
print("=" * 60)
print("ESTIMATED DELIVERY REPORT")
print("=" * 60)

print(f"Delivery Distance : {distance:.2f} km")
print(f"Preparation Time  : {new_order['Preparation_Time']} min")
print(f"Traffic Density   : {new_order['Road_traffic_density']}")
print(f"Driver Rating     : {new_order['Delivery_person_Ratings']:.1f}")

print("-" * 60)

print(f"Estimated Delivery Time : {prediction[0]:.2f} minutes")

print("=" * 60)

#tree_predictions = np.array([tree.predict(features)[0] for tree in model.estimators_])

#print(f"Average Prediction : {tree_predictions.mean():.2f} minutes")
#print(f"Prediction Std Dev : {tree_predictions.std():.2f} minutes")