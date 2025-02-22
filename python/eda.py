import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
import os
from typing import Dict, Any, List
from scipy.stats import shapiro
import joblib
import uvicorn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
import logging
# import torch
# from torch.cuda import is_available
# import cupy as cp
# from cuml import LinearRegression as cuLinearRegression
# from cuml import LogisticRegression as cuLogisticRegression
# from cuml import RandomForestClassifier as cuRandomForestClassifier
# from cuml import RandomForestRegressor as cuRandomForestRegressor
# from cuml.preprocessing import StandardScaler as cuStandardScaler
# from cuml.preprocessing import MinMaxScaler as cuMinMaxScaler

# Enable GPU if available
# DEVICE = "cuda" if is_available() else "cpu"
DEVICE = "cpu"
if DEVICE == "cuda":
    # Set default tensor type to cuda
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    pass

logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

# Enable CORS
origins = [
    "http://localhost:3000",  # Next.js app running locally
    "https://your-nextjs-app.vercel.app",  # Replace with your deployed Next.js app URL
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temporary directory to store processed files and models
TEMP_DIR = "temp"
MODEL_DIR = "models"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def identify_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Identify different types of columns in the dataset."""
    columns = {
        "numeric": df.select_dtypes(include=["float64", "int64"]).columns.tolist(),
        "categorical": df.select_dtypes(include=["object", "string"]).columns.tolist(),
        "text": []
    }
    
    # Identify text columns (columns with average word count > 3)
    for col in columns["categorical"]:
        if df[col].dtype == "object":
            avg_words = df[col].str.split().str.len().mean()
            if avg_words > 3:
                columns["text"].append(col)
                columns["categorical"].remove(col)
    
    return columns

def preprocess_text(text):
    """Basic text preprocessing function."""
    if pd.isna(text):
        return ""
    # Convert to string if not already
    text = str(text)
    # Convert to lowercase
    text = text.lower()
    # Remove special characters but keep spaces
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    return text

def determine_scaler(df: pd.DataFrame) -> str:
    """Determine the most suitable scaler based on numeric column distributions."""
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    if numeric_cols.empty:
        return "minmax"
    
    skewness = df[numeric_cols].skew().abs().mean()
    return "standard" if skewness < 1 else "minmax"

def convert_numpy_types(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert NumPy types to native Python types for JSON serialization."""
    converted_data = {}
    for key, value in data.items():
        if isinstance(value, dict):
            converted_data[key] = convert_numpy_types(value)
        elif isinstance(value, (np.integer, np.floating)):
            converted_data[key] = value.item()
        else:
            converted_data[key] = value
    return converted_data

@app.post("/process-csv/")
async def process_csv(file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        df = pd.read_csv(file.file)
        df.dropna(how="all", inplace=True)
        
        # Identify column types
        column_types = identify_column_types(df)
        
        # Process text columns
        text_features = {}
        for col in column_types["text"]:
            df[col] = df[col].apply(preprocess_text)
            # Create TF-IDF features for text columns
            tfidf = TfidfVectorizer(max_features=100)
            tfidf_matrix = tfidf.fit_transform(df[col].fillna(""))
            # Convert to DataFrame and add as new columns
            tfidf_df = pd.DataFrame(
                tfidf_matrix.toarray(),
                columns=[f"{col}_tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
            )
            df = pd.concat([df, tfidf_df], axis=1)
            # Store the vectorizer for later use
            text_features[col] = tfidf
            # Drop original text column as it's now encoded
            df.drop(columns=[col], inplace=True)
        
        # Save text feature extractors
        if text_features:
            joblib.dump(text_features, os.path.join(MODEL_DIR, "text_features.joblib"))
        
        # Process categorical columns
        for col in column_types["categorical"]:
            df[col] = df[col].str.lower()
            df[col] = df[col].str.replace(r"[^\w\s]", "", regex=True)
        
        # Handle missing values
        numeric_fill = {col: df[col].mean() for col in column_types["numeric"]}
        categorical_fill = {col: "unknown" for col in column_types["categorical"]}
        df.fillna(numeric_fill, inplace=True)
        df.fillna(categorical_fill, inplace=True)
        
        # Encode categorical variables
        if column_types["categorical"]:
            ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            df[column_types["categorical"]] = ordinal_encoder.fit_transform(df[column_types["categorical"]])
            joblib.dump(ordinal_encoder, os.path.join(MODEL_DIR, "ordinal_encoder.joblib"))
        
        # Scale numeric features
        scaler_type = determine_scaler(df)
        if column_types["numeric"]:
            if DEVICE == "cuda":
                # scaler = cuStandardScaler() if scaler_type == "standard" else cuMinMaxScaler()
                # # Convert to cupy array for GPU processing
                # numeric_data = cp.array(df[column_types["numeric"]])
                # df[column_types["numeric"]] = scaler.fit_transform(numeric_data)
                pass
            else:
                scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
                df[column_types["numeric"]] = scaler.fit_transform(df[column_types["numeric"]])
            joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
        
        # Save processed DataFrame
        output_path = os.path.join(TEMP_DIR, "processed_" + file.filename)
        df.to_csv(output_path, index=False)
        
        # Generate summary statistics
        summary_stats = {}
        for col in df.columns:
            stats = df[col].describe().to_dict()
            stats["missing_values"] = int(df[col].isnull().sum())
            summary_stats[col] = stats
        
        return {
            "message": f"CSV processed successfully with {scaler_type.capitalize()}Scaler and text processing",
            "summary_statistics": convert_numpy_types(summary_stats),
            "download_link": f"/download/{os.path.basename(output_path)}"
        }
    except Exception as e:
        logging.error(f"Error processing CSV: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/train-model/")
async def train_model(
    file: UploadFile = File(...),
    target_column: str = Form(...),
):
    try:
        logging.debug("Received file upload request.")
        
        # Validate file
        if not file.filename.endswith('.csv'):
            return JSONResponse(
                status_code=400,
                content={"error": "Only CSV files are supported"}
            )
            
        # Read and validate data
        try:
            df = pd.read_csv(file.file)
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"error": f"Error reading CSV file: {str(e)}"}
            )
        
        if df.empty:
            return JSONResponse(
                status_code=400,
                content={"error": "The uploaded file is empty"}
            )
            
        if target_column not in df.columns:
            return JSONResponse(
                status_code=400,
                content={"error": f"Target column '{target_column}' not found in dataset. Available columns: {', '.join(df.columns)}"}
            )

        # Prepare data
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Validate target variable
        if y.isnull().any():
            return JSONResponse(
                status_code=400,
                content={"error": "Target column contains missing values"}
            )
        
        # Determine if the task is classification or regression
        is_classification = y.dtype == "object" or y.nunique() <= 20
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize models
        models = {
            "logistic_regression": LogisticRegression() if is_classification else LinearRegression(),
            "decision_tree": DecisionTreeClassifier() if is_classification else DecisionTreeRegressor(),
        }

        # Train and evaluate models
        model_scores = {}
        best_model_name = None
        best_model = None
        best_model_score = float('-inf')  # We'll use negative MSE for regression too

        for model_name, model in models.items():
            logging.debug(f"Training model: {model_name}")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                if is_classification:
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average="weighted")
                    score = accuracy
                    model_scores[model_name] = {
                        "accuracy": round(float(accuracy), 4),
                        "f1_score": round(float(f1), 4)
                    }
                else:
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    score = -mse  # Negative MSE for consistency (higher is better)
                    model_scores[model_name] = {
                        "mean_squared_error": round(float(mse), 4),
                        "r2_score": round(float(r2), 4)
                    }
                
                # Update best model
                if score > best_model_score:
                    best_model_score = score
                    best_model_name = model_name
                    best_model = model
                    
            except Exception as model_error:
                logging.error(f"Error training {model_name}: {str(model_error)}")
                model_scores[model_name] = {"error": str(model_error)}

        # Ensure we found at least one working model
        if best_model is None:
            return JSONResponse(
                status_code=500,
                content={"error": "No models were successfully trained"}
            )

        # Save the best model
        try:
            best_model_filename = f"{best_model_name}_model.joblib"
            best_model_path = os.path.join(MODEL_DIR, best_model_filename)
            os.makedirs(MODEL_DIR, exist_ok=True)
            joblib.dump(best_model, best_model_path)
        except Exception as save_error:
            logging.error(f"Error saving model: {str(save_error)}")
            return JSONResponse(
                status_code=500,
                content={"error": "Error saving the model file"}
            )

        return {
            "best_model_name": best_model_name,
            "best_model_score": round(float(best_model_score), 4),
            "evaluation_metrics_all_models": model_scores,
            "best_model_download_link": f"/download-model/{best_model_filename}",
            "task_type": "classification" if is_classification else "regression",
            "device_used": DEVICE
        }

    except Exception as e:
        logging.error(f"Error during model training: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Unexpected error: {str(e)}"}
        )

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

@app.post("/test-model/")
async def test_model(
    model_file: UploadFile = File(...),
    test_data_file: UploadFile = File(...),
    target_column: str = Form(...),
):
    try:
        # Load the trained model
        model_path = os.path.join(TEMP_DIR, "uploaded_model.joblib")
        with open(model_path, "wb") as f:
            f.write(await model_file.read())
        model = joblib.load(model_path)

        # Load the test data
        test_df = pd.read_csv(test_data_file.file)
        
        if target_column not in test_df.columns:
            return JSONResponse(status_code=400, content={"error": f"Target column '{target_column}' not found in test dataset."})

        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]

        # Convert to GPU if using CUDA
        if DEVICE == "cuda":
            # X_test = cp.array(X_test)
            # y_test = cp.array(y_test)
            pass

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        is_classification = isinstance(y_test.iloc[0], (str, bool)) or y_test.nunique() <= 20
        
        # Convert predictions to CPU if using GPU
        if DEVICE == "cuda":
            # y_pred = cp.asnumpy(y_pred)
            # y_test = cp.asnumpy(y_test)
            pass

        if is_classification:
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            result = {
                "accuracy": round(float(accuracy), 4),
                "f1_score": round(float(f1), 4),
                "predictions": y_pred.tolist(),
                "device_used": DEVICE
            }
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            result = {
                "mean_squared_error": round(float(mse), 4),
                "r2_score": round(float(r2), 4),
                "predictions": y_pred.tolist(),
                "device_used": DEVICE
            }

        # Clean up temporary model file
        if os.path.exists(model_path):
            os.remove(model_path)

        return result

    except Exception as e:
        logging.error(f"Error testing the model: {str(e)}")
        if os.path.exists(model_path):
            os.remove(model_path)
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    logging.info(f"Starting FastAPI server with device: {DEVICE}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )