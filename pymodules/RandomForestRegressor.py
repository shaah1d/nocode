## These are the libraries necessary so don't touch them keep it as it is
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib as jb
# Load dataset
data = pd.read_csv('..')#Add path to your dataset

# Define target columns, the one you want to predict
target_columns = ['..']

# Separate features (X) and target variables (y)
X = data.drop(columns=target_columns)
y = data[target_columns]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calling Random Forest Regressor Model with Class Weighting
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

# Train the model for each target column individually
for target_col in target_columns:
    print(f"\nTraining Random Forest Regressor for target: {target_col}")

    # Select the specific target column
    y_train_single = y_train[target_col]
    y_test_single = y_test[target_col]

    # Fit the model
    model.fit(X_train, y_train_single)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test_single, y_pred)
    r2 = r2_score(y_test_single, y_pred)

    # Print results
    print("\n--- Evaluation Metrics ---")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}")


#Save your model with this command
jb.dump(model, 'Random Forest Model.pkl')