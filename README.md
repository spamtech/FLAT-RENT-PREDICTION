# FLAT-RENT-PREDICTION
This project demonstrates how to apply Multiple Linear Regression using scikit-learn to predict house prices based on multiple features such as size, number of bedrooms, and distance from the city center.
It includes data preprocessing, model training, evaluation, and visualization of results, along with an interactive prediction mode.

# Features
Reads housing data from a CSV file (housing_data_500.csv) or sample dictionary dataset.
Splits dataset into training and testing sets.
Trains a Linear Regression model using multiple independent variables:
Size_sqft → House size (in square feet)
Bedrooms → Number of bedrooms
Distance_km → Distance from city center (in km)
Evaluates the model with metrics:
MAE (Mean Absolute Error)
MSE (Mean Squared Error)
RMSE (Root Mean Squared Error)
R² Score (Coefficient of Determination)
Displays:
Actual vs Predicted scatter plot
Residual plot for diagnostic checking
Supports interactive input mode for custom predictions.

# Key Learnings
How to implement Multiple Linear Regression with scikit-learn
Importance of train-test split for evaluation
Model performance evaluation using error metrics
Visual inspection with diagnostic plots
Extending ML models with interactive prediction
