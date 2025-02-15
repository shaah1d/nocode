# main.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
import os
from typing import Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import shapiro
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import joblib
import uvicorn

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

@app.post("/process-csv/")
async def process_csv(file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        # Read the uploaded CSV file into a Pandas DataFrame
        df = pd.read_csv(file.file)
        # Basic EDA steps
        # 1. Drop rows with all null values
        df.dropna(how="all", inplace=True)
        # 2. Fill remaining null values with column mean (for numeric columns)
        for col in df.select_dtypes(include=["float64", "int64"]).columns:
            df[col].fillna(df[col].mean(), inplace=True)
        # Generate summary statistics before scaling
        summary_stats = df.describe().to_dict()
        # Determine the most suitable scaler
        scaler_type = determine_scaler(df)
        # Apply the selected scaler
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
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
        return {
            "message": f"CSV processed successfully with {scaler_type.capitalize()}Scaler",
            "summary_statistics": summary_stats,
            "download_link": f"/download/{os.path.basename(output_path)}"
        }
    except Exception as e:
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

import logging

logging.basicConfig(level=logging.DEBUG)

@app.post("/train-model/")
async def train_model(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    model_type: str = Form("linear_regression")
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
        
        # Split features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        logging.debug(f"Features shape: {X.shape}, Target shape: {y.shape}")
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logging.debug(f"Training set size: {X_train.shape}, Testing set size: {X_test.shape}")
        
        # Select and train the model
        if model_type == "linear_regression":
            model = LinearRegression()
        elif model_type == "decision_tree":
            model = DecisionTreeRegressor()
        elif model_type == "random_forest":
            model = RandomForestRegressor()
        elif model_type == "svm":
            model = SVR()
        else:
            logging.error(f"Unsupported model type: {model_type}")
            return JSONResponse(status_code=400, content={"error": f"Unsupported model type: {model_type}"})
        
        logging.debug(f"Training model of type: {model_type}")
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logging.debug(f"Evaluation metrics - MSE: {mse}, R2: {r2}")
        
        # Save the trained model
        model_filename = f"{model_type}_model.joblib"
        model_path = os.path.join(MODEL_DIR, model_filename)
        joblib.dump(model, model_path)
        logging.debug(f"Model saved at: {model_path}")
        
        return {
            "message": f"Model trained successfully ({model_type})",
            "evaluation_metrics": {
                "mean_squared_error": mse,
                "r2_score": r2
            },
            "model_download_link": f"/download-model/{model_filename}"
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