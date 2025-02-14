from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load the trained XGBoost model
import joblib
model = joblib.load('YOUR MODEL NAME HERE')

# Make predictions on test data
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
r2 = r2_score(y_test, y_pred)            # R² Score
mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error

# Print results
print("\n--- XGBoost Regressor Test Results ---")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")