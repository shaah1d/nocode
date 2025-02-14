from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib as jb
# Hyperparameter grid for KNN
param_grid = {
    'n_neighbors': [3, 5, 7, 9],  # Number of neighbors to consider. Smaller values make the model more sensitive to noise; larger values smooth out predictions.
    'weights': ['uniform', 'distance'],  # 'uniform' treats all neighbors equally, while 'distance' gives closer neighbors more weight.
    'p': [1, 2]  # p=1 for Manhattan distance, p=2 for Euclidean distance. Try experimenting with other values like p=3 for Minkowski distance.
}

# Initialize KNN Regressor
model = KNeighborsRegressor()

# Perform Grid Search for Hyperparameter Tuning
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)  # Fit on training data. Ensure features are scaled before using KNN, as it's distance-based.

# Best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("\n--- KNN Results ---")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")

# Note: If the model is slow, reduce n_neighbors or use a smaller dataset. Scaling features (e.g., StandardScaler) is crucial for KNN performance.

jb.dump(best_model,"KNN Tuned Model.joblib")