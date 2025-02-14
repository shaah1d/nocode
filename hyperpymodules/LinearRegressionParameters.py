from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib as jb
# Hyperparameter grid for Ridge Regression
param_grid = {
    'alpha': [0.01, 0.1, 1, 10, 100]  # Regularization strength. Smaller values allow more complexity; larger values penalize large coefficients more.
}

# Initialize Ridge Regression
model = Ridge(random_state=42)

# Perform Grid Search for Hyperparameter Tuning
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)  # Fit on training data. Ridge regression works well when features are standardized.

# Best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("\n--- Ridge Regression Results ---")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")

# Note: If the model underfits, try reducing alpha. If it overfits, increase alpha to apply stronger regularization.

jb.dump(best_model, "Linear Regression Tuned Model.joblib")