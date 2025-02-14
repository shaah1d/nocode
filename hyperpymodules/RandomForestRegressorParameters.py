from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib as jb
# Hyperparameter grid for Random Forest Regressor
param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees in the forest. Start with smaller values like 50 or 100, but you can increase it (e.g., 150, 200) for better performance.
    'max_depth': [3, 5, 10, None],  # Maximum depth of each tree. Increase this value (e.g., 15, 20) if your dataset is large and complex.
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node. Higher values reduce overfitting but may underfit if too high.
    'min_samples_leaf': [1, 2, 4]  # Minimum samples required at a leaf node. Increasing this value can reduce overfitting.
}

# Initialize Random Forest Regressor
model = RandomForestRegressor(random_state=42)

# Perform Grid Search for Hyperparameter Tuning
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)  # Fit on training data. Random Forest is robust to overfitting, but tuning helps improve accuracy.

# Best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("\n--- Random Forest Regressor Results ---")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")

# Note: If the model is slow, reduce n_estimators or max_depth. For better results, try adding more estimators (e.g., 250, 300).

jb.dump(best_model,"Random Forest Tuned.pkl")