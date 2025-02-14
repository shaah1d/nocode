from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib as jb
# Hyperparameter grid for Decision Tree Regressor
param_grid = {
    'max_depth': [3, 5, 10, None],  # Controls the maximum depth of the tree. Start with small values like 3 or 5, but you can increase it (e.g., 15, 20) if your dataset is large.
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node. Higher values prevent overfitting but may underfit if too high.
    'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required at a leaf node. Increasing this value can reduce overfitting.
}

# Initialize Decision Tree Regressor
model = DecisionTreeRegressor(random_state=42)

# Perform Grid Search for Hyperparameter Tuning
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)  # Fit on training data. You can increase cv=3 to cv=5 for more robust cross-validation.

# Best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error measures the average squared difference between predicted and actual values.
r2 = r2_score(y_test, y_pred)  # R² Score explains the proportion of variance in the target variable explained by the model.

# Print results
print("\n--- Decision Tree Regressor Results ---")
print(f"Best Parameters: {grid_search.best_params_}")  # These are the optimal hyperparameters found during tuning.
print(f"Mean Squared Error: {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")

# Note: If you find the model underfitting, try increasing max_depth or reducing min_samples_split/min_samples_leaf.

jb.dump(best_model,"DT Regressor Tuned Model.joblib")