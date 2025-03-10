import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
import os
from typing import Dict, Any
import joblib
import uvicorn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import traceback  # For detailed error logging

logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

# Enable CORS
origins = ["http://localhost:3000", "https://your-nextjs-app.vercel.app"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = "temp"
MODEL_DIR = "models"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def determine_scaler(df: pd.DataFrame, numeric_cols: list) -> str:
    """Determine the most suitable scaler based on numeric column distributions."""
    if numeric_cols.empty:  # Fixed: Use .empty instead of direct boolean check
        return "minmax"
    skewness = df[numeric_cols].skew().abs().mean()
    return "standard" if skewness < 1 else "minmax"

def determine_missing_strategy(df: pd.DataFrame, col: str) -> str:
    """Determine the best missing value strategy for a column."""
    missing_pct = df[col].isnull().mean()
    if missing_pct > 0.5:
        return "drop"
    if pd.api.types.is_numeric_dtype(df[col]):
        skewness = df[col].skew()
        if pd.isna(skewness) or abs(skewness) > 1:
            return "median"
        elif df[col].nunique() / len(df[col]) < 0.05:
            return "mode"
        else:
            return "mean"
    return "mode"

def determine_encoding_strategy(df: pd.DataFrame, col: str) -> str:
    """Determine the best encoding strategy for a categorical column."""
    cardinality = df[col].nunique()
    if cardinality == 0:  # Handle case where column has no unique values
        return "drop"
    if cardinality / len(df[col]) > 0.1 or cardinality > 50:
        return "onehot"
    return "ordinal"

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

def generate_visualizations(df: pd.DataFrame) -> Dict[str, str]:
    """Generate visualizations and return them as base64-encoded images."""
    visualizations = {}
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    categorical_cols = df.select_dtypes(exclude=["float64", "int64", "datetime64"]).columns

    for col in numeric_cols[:3]:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        visualizations[f"histogram_{col}"] = base64.b64encode(buf.getvalue()).decode("utf-8")

    for col in categorical_cols[:3]:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=col, data=df)
        plt.title(f"Count of {col}")
        plt.xticks(rotation=45)
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        visualizations[f"bar_{col}"] = base64.b64encode(buf.getvalue()).decode("utf-8")

    if len(numeric_cols) > 1:
        plt.figure(figsize=(8, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        visualizations["heatmap"] = base64.b64encode(buf.getvalue()).decode("utf-8")

    return visualizations

@app.post("/process-csv/")
async def process_data(file: UploadFile = File(...), include_visualizations: bool = Form(default=False)) -> Dict[str, Any]:
    try:
        # Load file based on type, parsing dates if a Date column exists
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file.file)
        elif file.filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file.file)
        elif file.filename.endswith(".json"):
            df = pd.read_json(file.file)
        else:
            return JSONResponse(status_code=400, content={"error": "Unsupported file format"})
        
        # Convert date columns if they exist
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        # Detect datetime columns
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.difference(datetime_cols)
        categorical_cols = df.select_dtypes(exclude=["float64", "int64", "datetime64"]).columns

        # Handle missing values dynamically
        fill_values = {}
        columns_to_drop = []
        for col in df.columns:
            strategy = determine_missing_strategy(df, col)
            logging.debug(f"Column {col}: Missing strategy = {strategy}")
            if strategy == "drop":
                columns_to_drop.append(col)
            elif strategy == "mean":
                fill_values[col] = df[col].mean()
            elif strategy == "median":
                fill_values[col] = df[col].median()
            elif strategy == "mode":
                fill_values[col] = df[col].mode()[0] if not df[col].mode().empty else pd.NaT if col in datetime_cols else "Unknown"

        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            logging.debug(f"Dropped columns: {columns_to_drop}")
        if fill_values:
            df = df.fillna(fill_values)
            logging.debug(f"Filled missing values: {fill_values}")

        # Update column lists after dropping
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.difference(datetime_cols)
        categorical_cols = df.select_dtypes(exclude=["float64", "int64", "datetime64"]).columns

        # Text normalization and encoding only for non-datetime categorical columns
        for col in list(categorical_cols):
            if col in datetime_cols:
                continue
            df[col] = df[col].astype(str).str.lower().str.replace(r"[^\w\s]", "", regex=True)
            encoding = determine_encoding_strategy(df, col)
            logging.debug(f"Column {col}: Encoding strategy = {encoding}")
            if encoding == "drop":
                df = df.drop(columns=[col])
            elif encoding == "onehot":
                encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                encoded_data = encoder.fit_transform(df[[col]])
                encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([col]))
                df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)
            else:
                encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                df[col] = encoder.fit_transform(df[[col]])

        # Dynamic scaling for numeric columns only
        scaler_type = determine_scaler(df, numeric_cols)
        scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
        if not numeric_cols.empty:  # Fixed: Use .empty instead of direct boolean check
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            logging.debug(f"Applied {scaler_type} scaling to numeric columns: {numeric_cols}")

        # Save processed DataFrame
        output_path = os.path.join(TEMP_DIR, f"processed_{file.filename}")
        df.to_csv(output_path, index=False)

        # Summary statistics
        summary_stats = {col: df[col].describe().to_dict() for col in numeric_cols}
        summary_stats.update({
            col: {
                "unique_values": df[col].nunique(),
                "most_frequent": df[col].mode()[0] if not df[col].mode().empty else "Unknown",
                "missing_values": int(df[col].isnull().sum())
            } for col in categorical_cols
        })
        if len(datetime_cols) > 0:
            summary_stats.update({
                col: {
                    "min": str(df[col].min()),
                    "max": str(df[col].max()),
                    "most_frequent": str(df[col].mode()[0]) if not df[col].mode().empty else "Unknown",
                    "missing_values": int(df[col].isnull().sum())
                } for col in datetime_cols
            })

        response = {
            "message": f"Data processed with {scaler_type.capitalize()}Scaler and dynamic encoding",
            "summary_statistics": convert_numpy_types(summary_stats),
            "download_link": f"/download/{os.path.basename(output_path)}"
        }
        if include_visualizations:
            response["visualizations"] = generate_visualizations(df)
        return response
    except Exception as e:
        logging.error(f"Error processing data: {str(e)}\n{traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/download/{file_name}")
async def download_file(file_name: str):
    file_path = os.path.join(TEMP_DIR, file_name)
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "File not found"})
    return FileResponse(file_path, media_type="text/csv", filename=file_name)

@app.post("/train-model/")
async def train_model(file: UploadFile = File(...), target_column: str = Form(...), include_metrics_plots: bool = Form(default=False)):
    try:
        # Read CSV without assumptions about date columns
        df = pd.read_csv(file.file)
        
        # Convert date columns if they exist
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            
        if target_column not in df.columns:
            return JSONResponse(status_code=400, content={"error": f"Target column '{target_column}' not found"})

        X, y = df.drop(columns=[target_column]), df[target_column]
        is_classification = y.dtype == "object" or y.nunique() <= 20

        # Get list of column types
        datetime_cols = X.select_dtypes(include=["datetime64"]).columns
        numeric_cols = X.select_dtypes(include=["float64", "int64"]).columns.difference(datetime_cols)
        categorical_cols = X.select_dtypes(exclude=["float64", "int64", "datetime64"]).columns

        # Create transformers list
        transformers = []
        if not categorical_cols.empty:  # Fixed: Use .empty instead of direct boolean check
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols))
        if not numeric_cols.empty:  # Fixed: Use .empty instead of direct boolean check
            transformers.append(("num", StandardScaler(), numeric_cols))

        # Handle the case where transformers might be empty
        if not transformers:
            return JSONResponse(status_code=400, content={"error": "No valid features found for preprocessing"})

        preprocessor = ColumnTransformer(transformers=transformers)

        models = {
            "random_forest": RandomForestClassifier() if is_classification else RandomForestRegressor(),
            "svm": SVC(probability=True) if is_classification else SVR(),
        }

        model_scores = {}
        best_model_name, best_model_score, best_model_path = None, float('-inf'), None

        for model_name, model in models.items():
            pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy" if is_classification else "r2")
            score = cv_scores.mean()

            if is_classification:
                # Handle potential errors in f1_score calculation
                try:
                    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                except Exception as e:
                    logging.warning(f"Error calculating f1_score: {str(e)}")
                    f1 = 0
                    
                model_scores[model_name] = {
                    "cv_accuracy": round(score, 4),
                    "f1_score": round(f1, 4),
                }
            else:
                model_scores[model_name] = {
                    "cv_r2": round(score, 4),
                    "mse": round(mean_squared_error(y_test, y_pred), 4),
                }

            # Fixed: Comparison for both classification and regression
            if score > best_model_score:
                best_model_score = score
                best_model_name = model_name
                best_model_path = os.path.join(MODEL_DIR, f"{model_name}_model.joblib")
                joblib.dump(pipeline, best_model_path)

        response = {
            "best_model_name": best_model_name,
            "best_model_score": round(best_model_score, 4),
            "evaluation_metrics_all_models": model_scores,
            "best_model_download_link": f"/download-model/{os.path.basename(best_model_path)}",
        }

        if include_metrics_plots and is_classification:
            cm = confusion_matrix(y_test, pipeline.predict(X_test))
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Confusion Matrix for {best_model_name}")
            buf = BytesIO()
            plt.savefig(buf, format="png")
            plt.close()
            response["confusion_matrix"] = base64.b64encode(buf.getvalue()).decode("utf-8")

        return response
    except Exception as e:
        logging.error(f"Error during model training: {str(e)}\n{traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/download-model/{model_name}")
async def download_model(model_name: str):
    file_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "Model not found"})
    return FileResponse(file_path, media_type="application/octet-stream", filename=model_name)

@app.post("/test-model/")
async def test_model(model_file: UploadFile = File(...), test_data_file: UploadFile = File(...), target_column: str = Form(...)):
    try:
        model_path = os.path.join(TEMP_DIR, "uploaded_model.joblib")
        with open(model_path, "wb") as f:
            f.write(await model_file.read())
        model = joblib.load(model_path)

        # Read CSV without assumptions about date columns
        test_df = pd.read_csv(test_data_file.file)
        
        # Convert date columns if they exist
        if "Date" in test_df.columns:
            test_df["Date"] = pd.to_datetime(test_df["Date"], errors="coerce")
            
        if target_column not in test_df.columns:
            return JSONResponse(status_code=400, content={"error": f"Target column '{target_column}' not found"})

        X_test, y_test = test_df.drop(columns=[target_column]), test_df[target_column]
        y_pred = model.predict(X_test)

        is_classification = isinstance(y_test.iloc[0], (str, bool)) or y_test.nunique() <= 20
        if is_classification:
            # Handle potential errors in f1_score calculation
            try:
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            except Exception as e:
                logging.warning(f"Error calculating f1_score: {str(e)}")
                f1 = 0
                
            result = {
                "accuracy": round(accuracy_score(y_test, y_pred), 4),
                "f1_score": round(f1, 4),
                "predictions": y_pred.tolist(),
            }
        else:
            result = {
                "mean_squared_error": round(mean_squared_error(y_test, y_pred), 4),
                "r2_score": round(r2_score(y_test, y_pred), 4),
                "predictions": y_pred.tolist(),
            }
        return result
    except Exception as e:
        logging.error(f"Error testing model: {str(e)}\n{traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, port=8000)