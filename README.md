# Tehran-Housing-Price-Prediction
Tehran Housing Price Prediction
Welcome to the Tehran Housing Price Prediction project! This repository contains the code and resources for predicting housing prices in Tehran using machine learning techniques.

Project Overview
This project aims to provide accurate predictions of housing prices in Tehran based on various features such as area, number of rooms, and amenities (parking, elevator, warehouse). The dataset used for this project was scraped from the Divar website and contains around 3,000 entries.

Features
Data Preprocessing: Cleaned and preprocessed data for accurate modeling.
Model Training: Various regression models including Ridge, Lasso, ElasticNet, DecisionTree, RandomForest, KNeighbors, and XGBoost.
Model Evaluation: Comparison of models based on RMSE and R² scores.
Web Application: An interactive Streamlit web app for users to input features and get price predictions.
Installation
To run this project locally, follow these steps:

Clone the repository:


Copy code
git clone https://github.com/yourusername/tehran-housing-price-prediction.git
cd tehran-housing-price-prediction
Create and activate a virtual environment:


Copy code
python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`
Install the required packages:


Copy code
pip install -r requirements.txt
Run the Streamlit app:


Copy code
streamlit run app.py
Usage
Data: Ensure the dataset (df_no_outlier.csv) is placed in the correct directory.
Web App: Use the Streamlit web interface to select features and predict housing prices.
Model Details
The following models are trained and evaluated:

Ridge Regression
Lasso Regression
ElasticNet Regression
DecisionTreeRegressor
RandomForestRegressor
KNeighborsRegressor
XGBoostRegressor
Results
The models are evaluated based on RMSE and R² scores for both training and testing datasets. A comparison of the models is provided in the form of visual plots.


Contributing
Contributions are welcome! Please create a pull request or open an issue to discuss your ideas.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgements
Data Source: Divar website
Libraries Used: Scikit-learn, XGBoost, Streamlit, Seaborn, Matplotlib
Feel free to modify this template to better fit your project's specifics. Ensure you replace placeholder URLs and file names with actual ones relevant to your repository.
