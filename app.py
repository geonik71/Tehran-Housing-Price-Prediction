import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_extras.let_it_rain import rain


# Load pre-trained models
model_names = ["Ridge Regression", "Lasso Regression", "ElasticNet Regression", "DecisionTreeRegressor", "RandomForestRegressor", "KNeighborsRegressor", "XGBoostRegressor"]
models = {
    "Ridge Regression": joblib.load('Ridge_best_model.joblib'),
    "Lasso Regression": joblib.load('Lasso_best_model.joblib'),
    "ElasticNet Regression": joblib.load('ElasticNet_best_model.joblib'),
    "DecisionTreeRegressor": joblib.load('DecisionTreeRegressor_best_model.joblib'),
    "RandomForestRegressor": joblib.load('RandomForestRegressor_best_model.joblib'),
    "KNeighborsRegressor": joblib.load('KNeighborsRegressor_best_model.joblib'),
    "XGBoostRegressor": joblib.load('XGBRegressor_best_model.joblib')
}

# Streamlit app interface
# Introduction Text
st.title("Tehran Housing Price Prediction")
st.markdown('<hr class="title-line">', unsafe_allow_html=True)
st.image('https://toursofiran.com/wp-content/uploads/2024/01/TEHRAN-768x520.jpg')
introduction = """

Welcome to our Tehran Housing Price Prediction app! This application is designed to provide insights and predictions on housing prices in Tehran using machine learning techniques.

Our dataset, comprising approximately 3,000 rows, was meticulously scraped from the Divar website and includes crucial information such as:
- **Area** (in square meters)
- **Number of rooms**
- **Presence of parking**
- **Presence of an elevator**
- **Presence of a warehouse**
- **Price**

By leveraging this data, we've developed a robust machine learning model that predicts housing prices based on the mentioned features. This app allows users to interactively explore the dataset, visualize the results, and make informed decisions about the housing market in Tehran.

Dive in to explore the predictions and gain a deeper understanding of the factors influencing housing prices in one of Iran's most dynamic cities.
"""

# Display the introduction text
st.markdown(introduction)
st.image(r'E:\python\Practical Machine Learning\Project_1\Models Comparision.png',caption='Comparison of Different Models Based on Train and Test R² Score and RMSE')
selected_model_name = st.selectbox("Choose Your Model According to Accuracy Scores:", model_names)

# Features 
df = pd.read_csv('df_no_outlier.csv')
address = list(df['Address'].astype(str).unique())  # Convert all values to strings
address.sort()  # Sort the address list

st.sidebar.header("Select Your Features")
area_meter = st.sidebar.slider('Area (square meters)', 30, 250)
number_room = st.sidebar.number_input('Number of Rooms', 0, 5)
Parking_dict = {0: "No", 1: "Yes"}
has_parking = st.sidebar.radio('Has Parking', options=list(Parking_dict.keys()), format_func=lambda x: Parking_dict[x])
Warehouse_dict = {0: "No", 1: "Yes"}
has_Warehouse = st.sidebar.radio('Has Warehouse', options=list(Warehouse_dict.keys()), format_func=lambda x: Warehouse_dict[x])
Elevator_dict = {0: "No", 1: "Yes"}
has_Elevator = st.sidebar.radio('Has Elevator', options=list(Elevator_dict.keys()), format_func=lambda x: Elevator_dict[x])
address_select = st.sidebar.selectbox("Select Your Address", address)

row_list = [area_meter, number_room, has_parking, has_Warehouse, has_Elevator]

# Create a binary vector for the address
address_vector = [1 if addr == address_select else 0 for addr in address]
row_list.extend(address_vector)

# Convert to numpy array and reshape
row_array = np.array(row_list).reshape(1, -1)
# Prediction
selected_model = models[selected_model_name]
prediction = selected_model.predict(row_array)

if st.button('Predict'):
    rounded_prediction = round(prediction[0], -5)
    st.error(f"The estimated house price is: {rounded_prediction:,} Toman")
    rain(
        emoji="🏡",
        font_size=54,
        falling_speed=5,
        animation_length="infinite",)
    
