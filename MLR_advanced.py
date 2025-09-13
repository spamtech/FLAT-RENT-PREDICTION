# Step 1: Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def Multiple_Linear_Regression(df, features, target, ask_input=False, test_size=0.2, random_state=42):
    # Step 2: Prepare data
    X = df[features]
    y = df[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Step 3: Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Step 4: Predict
    y_pred = model.predict(X_test)

    # Step 5: Evaluate
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    print("---- Model Performance ----")
    print(f"MAE : {mae:.2f}")
    print(f"MSE : {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²  : {r2:.3f}\n")

    print("---- Model Coefficients ----")
    print("Intercept:", model.intercept_)
    for feat, coef in zip(features, model.coef_):
        print(f"{feat}: {coef:.4f}")

    # Step 6: Plots (diagnostics)
    plt.figure(figsize=(10,5))
    plt.scatter(y_test, y_pred, alpha=0.7, edgecolors="black")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

    residuals = y_test - y_pred
    plt.figure(figsize=(10,5))
    plt.scatter(y_pred, residuals, alpha=0.7, edgecolors="black")
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

    # Step 7: Interactive prediction
    if ask_input:
        try:
            size = float(input("Enter Size_sqft: "))
            beds = int(input("Enter Bedrooms: "))
            dist = float(input("Enter Distance_km: "))
            new_data = pd.DataFrame([[size, beds, dist]], columns=features)
            pred_price = model.predict(new_data)[0]
            print(f"\nPredicted Price: ₹{pred_price:.2f} Lakhs")
        except Exception as e:
            print(f"[Input skipped] {e}")

    return model

# Step 1: Dataset
'''data = {
    "Size_sqft": [1000, 1500, 2000, 2500, 3000, 1800, 2200, 2700, 3200, 3500],
    "Bedrooms": [2, 3, 3, 4, 4, 2, 3, 4, 5, 5],
    "Distance_km": [15, 10, 8, 5, 3, 12, 7, 6, 4, 2],
    "Price": [45, 65, 85, 120, 150, 95, 110, 135, 160, 180]
}'''
df=pd.read_csv("C:\\Users\\Dipnarayan\\OneDrive\\Desktop\\SKILL_GROWING\\AI_&_ML\\DAY_8\\housing_data_500.csv")
#df = pd.DataFrame(data)

# Step 2: Run
features = ["Size_sqft", "Bedrooms", "Distance_km"]
target = "Price"

model = Multiple_Linear_Regression(df, features, target, ask_input=True)