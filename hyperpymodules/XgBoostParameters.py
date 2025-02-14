from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib as jb
# Hyperparameter grid for XGBoost Regressor
param_grid = {
    'n_estimators': [50, 100, 200],  # Number of boosting rounds. Start with smaller values like 50 or 100, but you can increase it (e.g., 250, 300) for better performance.
    'learning_rate': [0.01, 0.1, 0.2],  # Step size shrinkage. Smaller values require more boosting rounds but can lead to better generalization.
    'max_depth': [3, 5, 7],  # Maximum depth of each tree. Increase this value (e.g., 9, 10) if your dataset is complex.
    'subsample': [0.8, 1.0],  # Fraction of samples used for each boosting round. Lower values reduce overfitting but may underfit if too low.
    'colsample_bytree': [0.8, 1.0]  # Fraction of features used for each tree. Lower values reduce overfitting but may underfit if too low.
}

# Initialize XGBoost Regressor
model = XGBRegressor(random_state=42)

# Perform Grid Search for Hyperparameter Tuning
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)  # Fit on training data. XGBoost is highly efficient and works well with large datasets.

# Best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("\n--- XGBoost Regressor Results ---")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")

# Note: If the model is slow, reduce n_estimators or learning_rate. For better results, try fine-tuning subsample and colsample_bytree.


#Save your model with this command
jb.dump(best_model, 'Decision Tree Tuned Model.pkl')