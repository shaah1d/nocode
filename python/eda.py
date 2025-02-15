# main.py
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
import os
from typing import Dict, Any
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import shapiro
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.svm import SVR
import joblib
import uvicorn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
import logging
from joblib import Parallel, delayed

logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

# Enable CORS
origins = [
    "http://localhost:3000",  # Next.js app running locally
    "https://your-nextjs-app.vercel.app",  # Replace with your deployed Next.js app URL
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Temporary directory to store processed files and models
TEMP_DIR = "temp"
MODEL_DIR = "models"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def determine_scaler(df: pd.DataFrame) -> str:
    """
    Automatically determine the most suitable scaler based on the dataset's characteristics.
    """
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    normality_scores = []
    for col in numeric_cols:
        # Perform Shapiro-Wilk test for normality
        stat, p_value = shapiro(df[col].dropna())
        normality_scores.append(p_value > 0.05)  # True if normal, False otherwise
    # Count how many columns are normally distributed
    normal_count = sum(normality_scores)
    total_numeric_cols = len(numeric_cols)
    # If more than 50% of numeric columns are normally distributed, use StandardScaler
    if normal_count / total_numeric_cols > 0.5:
        return "standard"
    else:
        return "minmax"


def convert_numpy_types(data):
    if isinstance(data, dict):
        return {k: convert_numpy_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(item) for item in data]
    elif isinstance(data, (np.int64, np.int32)):
        return int(data)
    elif isinstance(data, (np.float64, np.float32)):
        return float(data)
    else:
        return data

@app.post("/process-csv/")
async def process_csv(file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        # Read the uploaded CSV file into a Pandas DataFrame
        df = pd.read_csv(file.file)
        
        # Basic EDA steps
        # 1. Drop rows with all null values
        df.dropna(how="all", inplace=True)
        
        # 2. Fill remaining null values
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        categorical_cols = df.select_dtypes(include=["object", "string"]).columns
        
        fill_values = {
            col: df[col].mean() for col in numeric_cols
        }
        fill_values.update({col: "Unknown" for col in categorical_cols})
        df.fillna(value=fill_values, inplace=True)
        
        # Generate summary statistics for all columns
        summary_stats = {}
        for col in df.columns:
            if col in numeric_cols:
                # Numeric column: use describe()
                stats = df[col].describe().to_dict()
                stats["missing_values"] = int(df[col].isnull().sum())  # Count missing values
                summary_stats[col] = stats
            elif col in categorical_cols:
                # Categorical column: count unique values and most frequent category
                value_counts = df[col].value_counts().to_dict()
                summary_stats[col] = {
                    "unique_values": len(value_counts),
                    "most_frequent": max(value_counts, key=value_counts.get),
                    "most_frequent_count": max(value_counts.values()),
                    "missing_values": int(df[col].isnull().sum()),  # Count missing values
                }
        
        # Determine the most suitable scaler
        scaler_type = determine_scaler(df)
        
        # Apply the selected scaler
        if scaler_type == "standard":
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        elif scaler_type == "minmax":
            scaler = MinMaxScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        # Save the processed DataFrame to a new CSV file
        output_path = os.path.join(TEMP_DIR, "processed_" + file.filename)
        df.to_csv(output_path, index=False)
        
        # Verify the file exists
        if not os.path.exists(output_path):
            return JSONResponse(status_code=500, content={"error": "Failed to save processed file"})
        
        # Convert NumPy types to native Python types
        summary_stats = convert_numpy_types(summary_stats)
        
        return {
            "message": f"CSV processed successfully with {scaler_type.capitalize()}Scaler",
            "summary_statistics": summary_stats,
            "download_link": f"/download/{os.path.basename(output_path)}"
        }
    except Exception as e:
        logging.error(f"Error processing CSV: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/download/{file_name}")
async def download_file(file_name: str):
    file_path = os.path.join(TEMP_DIR, file_name)
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "File not found"})
    return FileResponse(
        file_path,
        media_type="text/csv",
        filename=file_name,
        headers={"Content-Disposition": f"attachment; filename={file_name}"}
    )



@app.post("/train-model/")
async def train_model(
    file: UploadFile = File(...),
    target_column: str = Form(...),
):
    try:
        logging.debug("Received file upload request.")
        # Load the processed CSV file
        df = pd.read_csv(file.file)
        logging.debug(f"Loaded CSV with columns: {df.columns.tolist()}")

        # Ensure the target column exists
        if target_column not in df.columns:
            logging.error(f"Target column '{target_column}' not found in dataset.")
            return JSONResponse(status_code=400, content={"error": f"Target column '{target_column}' not found in dataset."})

        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Determine if the task is classification or regression
        is_classification = y.dtype == "object" or y.nunique() <= 20  # Treat numeric columns with <= 20 unique values as classification

        # Identify text and numeric columns
        text_columns = X.select_dtypes(include=["object", "string"]).columns
        numeric_columns = X.select_dtypes(include=["float64", "int64"]).columns

        # Preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ("text", TfidfVectorizer(), text_columns[0]) if len(text_columns) > 0 else ("text", "passthrough", []),
                ("num", StandardScaler(), numeric_columns),
            ]
        )

        # Define the models to train
        models = {
            "logistic_regression": LogisticRegression(),
            "decision_tree_classifier": DecisionTreeClassifier(),
            "random_forest_classifier": RandomForestClassifier(),
            "svm_classifier": SVC(),
            "linear_regression": LinearRegression(),
            "decision_tree_regressor": DecisionTreeRegressor(),
            "random_forest_regressor": RandomForestRegressor(),
            "svm_regressor": SVR(),
        }

        # Select appropriate models based on the task
        selected_models = (
            {k: v for k, v in models.items() if "classifier" in k}
            if is_classification
            else {k: v for k, v in models.items() if "regressor" in k or "linear_regression" in k}
        )

        # Dictionary to store evaluation metrics for each model
        model_scores = {}
        best_model_name = None
        best_model_score = float('-inf') if is_classification else float('inf')
        best_model_path = None

        # Train and evaluate each model
        for model_name, model in selected_models.items():
            logging.debug(f"Training model of type: {model_name}")
            # Create a pipeline with preprocessing and the model
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model),
            ])

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train the model
            pipeline.fit(X_train, y_train)

            # Evaluate the model
            y_pred = pipeline.predict(X_test)
            if is_classification:
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")
                score = accuracy  # Use accuracy as the primary metric
                model_scores[model_name] = {
                    "accuracy": round(accuracy, 4),
                    "f1_score": round(f1, 4),
                }
            else:
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                score = mse  # Use MSE as the primary metric
                model_scores[model_name] = {
                    "mean_squared_error": round(mse, 4),
                    "r2_score": round(r2, 4),
                }

            # Check if this is the best model so far
            if (is_classification and score > best_model_score) or (not is_classification and score < best_model_score):
                best_model_score = score
                best_model_name = model_name
                best_model_path = os.path.join(MODEL_DIR, f"{model_name}_model.joblib")
                joblib.dump(pipeline, best_model_path)  # Save the best model

        # Return the results
        return {
            "best_model_name": best_model_name,
            "best_model_score": round(best_model_score, 4),
            "evaluation_metrics_all_models": model_scores,
            "best_model_download_link": f"/download-model/{os.path.basename(best_model_path)}",
        }
    except Exception as e:
        logging.error(f"Error during model training: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/download-model/{model_name}")
async def download_model(model_name: str):
    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        return JSONResponse(status_code=404, content={"error": "Model not found"})
    return FileResponse(
        model_path,
        media_type="application/octet-stream",
        filename=model_name,
        headers={"Content-Disposition": f"attachment; filename={model_name}"}
    )

if __name__ == "__main__":
    uvicorn.run(app, port=8000)