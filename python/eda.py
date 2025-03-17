from datetime import datetime
import json
import uuid
import numpy as np
from fastapi import FastAPI, File, Request, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
import os
from typing import Dict, Any, Optional
import joblib
import uvicorn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.model_selection import GridSearchCV, RepeatedKFold, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor, StackingClassifier, StackingRegressor
from sklearn.svm import SVC, SVR
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import traceback
from imblearn.over_sampling import SMOTE 

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Initialize FastAPI app
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

# Define directories
TEMP_DIR = "temp"
MODEL_DIR = "models"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Helper Functions
def determine_scaler(df: pd.DataFrame, numeric_cols: pd.Index) -> str:
    """Determine the most suitable scaler based on numeric column distributions."""
    if numeric_cols.empty:
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
    if cardinality == 0:
        return "drop"
    if cardinality / len(df[col]) > 0.1 or cardinality > 50:
        return "onehot"
    return "ordinal"

def convert_numpy_types(data: Any) -> Any:
    """Recursively convert NumPy types to native Python types for JSON serialization."""
    if isinstance(data, dict):
        return {key: convert_numpy_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(item) for item in data]
    elif isinstance(data, (np.integer, np.floating)):
        return data.item()
    elif isinstance(data, np.ndarray):
        return data.tolist()
    return data

def generate_visualizations(df: pd.DataFrame) -> Dict[str, str]:
    """Generate visualizations and return them as base64-encoded images."""
    visualizations = {}
    numeric_cols = df.select_dtypes(include=["number"]).columns
    categorical_cols = df.select_dtypes(exclude=["number", "datetime64"]).columns

    if not numeric_cols.empty:
        for col in numeric_cols[:3]:
            plt.figure(figsize=(6, 4))
            sns.histplot(df[col], kde=True)
            plt.title(f"Distribution of {col}")
            buf = BytesIO()
            plt.savefig(buf, format="png")
            plt.close()
            visualizations[f"histogram_{col}"] = base64.b64encode(buf.getvalue()).decode("utf-8")

    if not categorical_cols.empty:
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

def handle_outliers(df: pd.DataFrame, numeric_cols: pd.Index) -> pd.DataFrame:
    """Cap outliers using the IQR method for each numeric column."""
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower_bound, upper_bound)
    return df

def engineer_datetime_features(df: pd.DataFrame, datetime_cols: pd.Index) -> pd.DataFrame:
    """Extract features from datetime columns and drop the original columns."""
    for col in datetime_cols:
        df[f"{col}_year"] = df[col].dt.year
        df[f"{col}_month"] = df[col].dt.month
        df[f"{col}_day"] = df[col].dt.day
        df[f"{col}_weekday"] = df[col].dt.weekday
    # Drop original datetime columns after feature extraction
    df = df.drop(columns=datetime_cols)
    return df

def remove_highly_correlated_features(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """Remove features that are highly correlated to reduce redundancy."""
    numeric_cols = df.select_dtypes(include=["number"]).columns
    corr_matrix = df[numeric_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    if to_drop:
        df = df.drop(columns=to_drop)
    return df

# API Endpoint
@app.post("/process-csv/",
          summary="Process uploaded dataset",
          response_description="Processed data summary and download link")
async def process_data(
    request: Request,
    file: UploadFile = File(...),
    include_visualizations: bool = Form(default=False),
    target_column: str = Form(default=None)
) -> Dict[str, Any]:
    """
    Process an uploaded dataset and return a downloadable processed CSV.
    
    - **file**: CSV, Excel, or JSON file containing the dataset
    - **include_visualizations**: Whether to generate and return data visualizations
    - **target_column**: Name of the target variable column
    
    Returns processed data with automatic feature engineering applied.
    """
    try:
        # Validate file size before processing
        file_size = 0
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)
        
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            return JSONResponse(status_code=413, content={"error": "File too large (>100MB)"})
            
        # Load file based on type
        if file.filename.endswith(".csv"):
            if file_size > 10 * 1024 * 1024:  # 10MB
                chunk_size = 10000  # rows
                chunks = pd.read_csv(file.file, chunksize=chunk_size)
                df = pd.concat([chunk for chunk in chunks])
            else:
                df = pd.read_csv(file.file)
        elif file.filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file.file)
        elif file.filename.endswith(".json"):
            df = pd.read_json(file.file)
        else:
            return JSONResponse(status_code=400, content={"error": "Unsupported file format"})

        # Convert 'Date' column if it exists
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        
        # Separate target column if provided
        if target_column and target_column in df.columns:
            target = df[target_column]
            df = df.drop(columns=[target_column])
        else:
            target = None

        # Generate summary statistics before transformations
        summary_stats = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                summary_stats[col] = convert_numpy_types(df[col].describe().to_dict())
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                summary_stats[col] = {
                    "min": str(df[col].min()),
                    "max": str(df[col].max()),
                    "most_frequent": str(df[col].mode()[0]) if not df[col].mode().empty else "Unknown",
                    "missing_values": int(df[col].isnull().sum())
                }
            else:
                summary_stats[col] = {
                    "unique_values": df[col].nunique(),
                    "most_frequent": df[col].mode()[0] if not df[col].mode().empty else "Unknown",
                    "missing_values": int(df[col].isnull().sum())
                }

        # Identify column types in features
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns
        numeric_cols = df.select_dtypes(include=["number"]).columns
        categorical_cols = df.select_dtypes(exclude=["number", "datetime64"]).columns

        # Handle missing values in features
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
                fill_values[col] = df[col].mode()[0] if not df[col].mode().empty else (pd.NaT if col in datetime_cols else "Unknown")
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            logging.debug(f"Dropped columns: {columns_to_drop}")
        if fill_values:
            df = df.fillna(fill_values)
            logging.debug(f"Filled missing values: {fill_values}")

        # Update column lists after missing value handling
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns
        numeric_cols = df.select_dtypes(include=["number"]).columns
        categorical_cols = df.select_dtypes(exclude=["number", "datetime64"]).columns

        # Handle outliers on numeric features
        if not numeric_cols.empty:
            df = handle_outliers(df, numeric_cols)
            logging.debug("Outliers handled using IQR capping.")

        # Engineer datetime features and drop original datetime columns
        if not datetime_cols.empty:
            df = engineer_datetime_features(df, datetime_cols)
            logging.debug("Datetime features engineered and original datetime columns dropped.")

        # Remove highly correlated features
        df = remove_highly_correlated_features(df, threshold=0.95)
        logging.debug("Removed highly correlated features.")

        # Update feature column lists after removal
        numeric_cols = df.select_dtypes(include=["number"]).columns
        categorical_cols = df.select_dtypes(exclude=["number"]).columns
        scaler_type = determine_scaler(df, numeric_cols)

        preprocessing_metadata = {
            "missing_value_strategy": {col: determine_missing_strategy(df, col) for col in df.columns},
            "encoding_strategy": {col: determine_encoding_strategy(df, col) for col in categorical_cols},
            "scaler_type": scaler_type,
            "removed_columns": columns_to_drop,
            "timestamp": datetime.now().isoformat()
        }

        # Encode categorical features
        for col in list(categorical_cols):
            # Clean text values
            df[col] = df[col].astype(str).str.lower().str.replace(r"[^\w\s]", "", regex=True)
            encoding = determine_encoding_strategy(df, col)
            logging.debug(f"Column {col}: Encoding strategy = {encoding}")
            if encoding == "drop":
                df = df.drop(columns=[col])
            elif encoding == "onehot":
                encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                encoded_data = encoder.fit_transform(df[[col]])
                encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([col]), index=df.index)
                df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)
            else:  # ordinal
                encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                df[col] = encoder.fit_transform(df[[col]])

        # Update numeric columns after encoding (include newly created onehot features)
        numeric_cols = df.select_dtypes(include=["number"]).columns

        # Scale numeric features
        
        scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
        if not numeric_cols.empty:
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            logging.debug(f"Applied {scaler_type} scaling to numeric features: {numeric_cols}")

        # Process target column if provided
        if target is not None:
            # If target is non-numeric, encode it using ordinal encoding
            if not pd.api.types.is_numeric_dtype(target):
                target = target.astype(str).str.lower().str.replace(r"[^\w\s]", "", regex=True)
                target = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit_transform(target.values.reshape(-1, 1)).flatten()
            # Reattach target column to the dataset
            df[target_column] = target

        # If target exists and is binary, apply SMOTE to balance the dataset
        if target is not None and target_column in df.columns:
            y = df[target_column]
            if y.nunique() == 2:
                X = df.drop(columns=[target_column])
                class_counts = y.value_counts()
                imbalance_ratio = class_counts.min() / class_counts.max()
                if imbalance_ratio < 0.5:
                    smote = SMOTE()
                    X_res, y_res = smote.fit_resample(X, y)
                    df = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.DataFrame(y_res, columns=[target_column])], axis=1)
                    logging.debug("Applied SMOTE to balance the dataset.")

        # Save processed DataFrame
        output_path = os.path.join(TEMP_DIR, f"processed_{file.filename}")
        df.to_csv(output_path, index=False)

        # Record preprocessing steps for metadata
        
        
        # Save preprocessing metadata
        metadata_path = os.path.join(TEMP_DIR, f"metadata_{file.filename}.json")
        with open(metadata_path, 'w') as f:
            json.dump(convert_numpy_types(preprocessing_metadata), f)

        # Prepare response
        response = {
            "message": f"Data processed with {scaler_type.capitalize()} scaling and dynamic encoding.",
            "summary_statistics": summary_stats,
            "preprocessing_metadata": preprocessing_metadata,
            "download_link": f"/download/{os.path.basename(output_path)}"
        }
        if include_visualizations:
            response["visualizations"] = generate_visualizations(df)
        return response
    except Exception as e:
        logging.error(f"Error processing data: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/download/{file_name}")
async def download_file(file_name: str):
    """Download a processed file."""
    file_path = os.path.join(TEMP_DIR, file_name)
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "File not found"})
    return FileResponse(file_path, media_type="text/csv", filename=file_name)

@app.post("/train-model/",
          summary="Train ML model on processed dataset",
          response_description="Model metrics and download link")
async def train_model(
    request: Request,
    file: UploadFile = File(...),
    target_column: str = Form(...),
    include_metrics_plots: bool = Form(default=False)
):
    """
    Train a model on the processed dataset.
    
    - **file**: Processed CSV file
    - **target_column**: Name of the target variable column
    - **include_metrics_plots**: Whether to generate and return model evaluation plots
    
    Returns model evaluation metrics and a link to download the trained model.
    """
    try:
        request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
        logging.info(f"Request {request_id}: Starting model training")
        
        df = pd.read_csv(file.file)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        if target_column not in df.columns:
            return JSONResponse(status_code=400, content={"error": f"Target column '{target_column}' not found"})

        X, y = df.drop(columns=[target_column]), df[target_column]

        # Determine problem type (classification or regression)
        is_classification = False
        if pd.api.types.is_object_dtype(y):
            is_classification = True
        elif pd.api.types.is_integer_dtype(y) and y.nunique() <= 10:
            is_classification = True
        elif pd.api.types.is_float_dtype(y):
            is_classification = False

        logging.info(f"Request {request_id}: Target column '{target_column}' classified as: {'Classification' if is_classification else 'Regression'}")
        logging.info(f"Request {request_id}: Target data type: {y.dtype}, Number of unique values: {y.nunique()}")

        # Identify column types in features
        datetime_cols = X.select_dtypes(include=["datetime64"]).columns
        numeric_cols = X.select_dtypes(include=["float64", "int64"]).columns.difference(datetime_cols)
        categorical_cols = X.select_dtypes(exclude=["float64", "int64", "datetime64"]).columns

        # Build transformers for preprocessing
        transformers = []
        if not categorical_cols.empty:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols))
        if not numeric_cols.empty:
            transformers.append(("num", StandardScaler(), numeric_cols))
        if not transformers:
            return JSONResponse(status_code=400, content={"error": "No valid features found for preprocessing"})

        preprocessor = ColumnTransformer(transformers=transformers)

        # Define models and parameter grids based on problem type
        if is_classification:
            # Ensure target is categorical
            y = y.astype('category')
            models = {
                "random_forest": (RandomForestClassifier(), {
                    "model__n_estimators": [50, 100],
                    "model__max_depth": [None, 10, 20]
                }),
                "svc": (SVC(probability=True), {
                    "model__C": [0.1, 1, 10],
                    "model__kernel": ["rbf", "linear"]
                }),
                "gradient_boosting": (GradientBoostingClassifier(), {
                    "model__n_estimators": [50, 100],
                    "model__learning_rate": [0.01, 0.1]
                })
            }
            scoring_metric = "accuracy"
        else:
            models = {
                "random_forest": (RandomForestRegressor(), {
                    "model__n_estimators": [50, 100],
                    "model__max_depth": [None, 10, 20]
                }),
                "svr": (SVR(), {
                    "model__C": [0.1, 1, 10],
                    "model__epsilon": [0.01, 0.1, 1]
                }),
                "gradient_boosting": (GradientBoostingRegressor(), {
                    "model__n_estimators": [50, 100],
                    "model__learning_rate": [0.01, 0.1]
                })
            }
            scoring_metric = "r2"

        model_scores = {}
        best_model_name, best_model_score, best_model_path, best_pipeline = None, float("-inf"), None, None

        # Use repeated K-Fold for a more robust estimate
        cv_strategy = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)

        # Loop through models and tune hyperparameters using GridSearchCV
        for model_name, (model, param_grid) in models.items():
            # Split data before applying pipeline to prevent data leakage
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
            grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=cv_strategy, scoring=scoring_metric)
            
            try:
                grid_search.fit(X_train, y_train)
                best_estimator = grid_search.best_estimator_
                y_pred = best_estimator.predict(X_test)
                cv_score = grid_search.best_score_

                if is_classification:
                    try:
                        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                    except Exception as ex:
                        logging.warning(f"Error calculating f1_score: {str(ex)}")
                        f1 = 0
                    model_scores[model_name] = {
                        "cv_accuracy": round(cv_score, 4),
                        "f1_score": round(f1, 4)
                    }
                else:
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    model_scores[model_name] = {
                        "cv_r2": round(cv_score, 4),
                        "mse": round(mse, 4),
                        "r2": round(r2, 4)
                    }

                # Update best model if current model performs better
                if cv_score > best_model_score:
                    best_model_score = cv_score
                    best_model_name = model_name
                    best_pipeline = best_estimator
                    best_model_path = os.path.join(MODEL_DIR, f"{model_name}_model.joblib")
                    joblib.dump(best_pipeline, best_model_path)

                # Add feature importance if available
                if hasattr(best_estimator[-1], 'feature_importances_'):
                    try:
                        feature_importances = best_estimator[-1].feature_importances_
                        feature_names = X.columns.tolist()
                        # Try to get transformed feature names if available
                        if hasattr(best_estimator[0], 'get_feature_names_out'):
                            try:
                                feature_names = best_estimator[0].get_feature_names_out()
                            except Exception as e:
                                logging.warning(f"Could not get feature names: {str(e)}")
                        
                        importances = dict(zip(feature_names, feature_importances))
                        sorted_importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
                        model_scores[model_name]["feature_importances"] = sorted_importances
                    except Exception as e:
                        logging.warning(f"Could not calculate feature importances: {str(e)}")

            except Exception as e:
                logging.error(f"Error fitting {model_name}: {str(e)}")
                model_scores[model_name] = {"error": str(e)}
                continue

        if best_model_name is None:
            return JSONResponse(status_code=500, content={"error": "All models failed to train"})
        
        # Create model metadata
        model_metadata = {
            "creation_date": datetime.now().isoformat(),
            "feature_columns": X.columns.tolist(),
            "target_column": target_column,
            "problem_type": "classification" if is_classification else "regression",
            "metrics": model_scores[best_model_name],
            "preprocessing_steps": str(transformers),
            "best_params": grid_search.best_params_
        }
        
        # Save metadata alongside model
        model_metadata_path = os.path.join(MODEL_DIR, f"{best_model_name}_metadata.json")
        with open(model_metadata_path, 'w') as f:
            json.dump(convert_numpy_types(model_metadata), f)

        response = {
            "best_model_name": best_model_name,
            "best_model_score": round(best_model_score, 4),
            "problem_type": "classification" if is_classification else "regression",
            "evaluation_metrics_all_models": convert_numpy_types(model_scores),
            "best_model_download_link": f"/download-model/{os.path.basename(best_model_path)}",
            "model_metadata_link": f"/model-metadata/{os.path.basename(model_metadata_path)}"
        }

        # Include additional metric plots
        if include_metrics_plots:
            if is_classification:
                cm = confusion_matrix(y_test, best_pipeline.predict(X_test))
                plt.figure(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                plt.title(f"Confusion Matrix for {best_model_name}")
                buf = BytesIO()
                plt.savefig(buf, format="png")
                plt.close()
                response["confusion_matrix"] = base64.b64encode(buf.getvalue()).decode("utf-8")
            else:
                # Scatter plot of predicted vs actual values for regression
                plt.figure(figsize=(6, 4))
                plt.scatter(y_test, best_pipeline.predict(X_test), alpha=0.7)
                plt.xlabel("Actual")
                plt.ylabel("Predicted")
                plt.title(f"Actual vs Predicted for {best_model_name}")
                buf = BytesIO()
                plt.savefig(buf, format="png")
                plt.close()
                response["predicted_vs_actual"] = base64.b64encode(buf.getvalue()).decode("utf-8")

        return response

    except Exception as e:
        logging.error(f"Error during model training: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})
    

@app.post("/tune-model/")
async def tune_model(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    include_metrics_plots: bool = Form(default=False)
) -> Dict[str, Any]:
    """
    Perform extensive hyperparameter tuning and ensemble stacking to improve model accuracy.
    Returns the best model details along with evaluation statistics and optional metric plots.
    """
    try:
        df = pd.read_csv(file.file)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        if target_column not in df.columns:
            return JSONResponse(status_code=400, content={"error": f"Target column '{target_column}' not found"})

        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Determine problem type (classification vs. regression)
        is_classification = False
        if pd.api.types.is_object_dtype(y):
            is_classification = True
        elif pd.api.types.is_integer_dtype(y) and y.nunique() <= 10:
            is_classification = True
        elif pd.api.types.is_float_dtype(y):
            is_classification = False

        logging.info(f"Target column '{target_column}' identified as: {'Classification' if is_classification else 'Regression'}")
        logging.info(f"Target dtype: {y.dtype}, Unique values: {y.nunique()}")

        # Identify feature column types
        datetime_cols = X.select_dtypes(include=["datetime64"]).columns
        numeric_cols = X.select_dtypes(include=["float64", "int64"]).columns.difference(datetime_cols)
        categorical_cols = X.select_dtypes(exclude=["float64", "int64", "datetime64"]).columns

        # Create a preprocessing transformer
        transformers = []
        if not categorical_cols.empty:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols))
        if not numeric_cols.empty:
            transformers.append(("num", StandardScaler(), numeric_cols))
        if not transformers:
            return JSONResponse(status_code=400, content={"error": "No valid features found for preprocessing"})

        preprocessor = ColumnTransformer(transformers=transformers)

        # For classification, ensure target is categorical
        if is_classification:
            y = y.astype("category")
        
        # Define candidate models and hyperparameter grids
        if is_classification:
            candidate_models = {
                "random_forest": (
                    RandomForestClassifier(),
                    {"model__n_estimators": [50, 100, 200],
                     "model__max_depth": [None, 10, 20]}
                ),
                "svc": (
                    SVC(probability=True),
                    {"model__C": [0.1, 1, 10],
                     "model__kernel": ["rbf", "linear"]}
                ),
                "gradient_boosting": (
                    GradientBoostingClassifier(),
                    {"model__n_estimators": [50, 100],
                     "model__learning_rate": [0.01, 0.1]}
                ),
                "stacking": (
                    StackingClassifier(
                        estimators=[
                            ("rf", RandomForestClassifier()),
                            ("gb", GradientBoostingClassifier())
                        ],
                        final_estimator=SVC(probability=True)
                    ),
                    {}  # You can add hyperparameters for stacking if needed.
                )
            }
            scoring_metric = "accuracy"
        else:
            candidate_models = {
                "random_forest": (
                    RandomForestRegressor(),
                    {"model__n_estimators": [50, 100, 200],
                     "model__max_depth": [None, 10, 20]}
                ),
                "svr": (
                    SVR(),
                    {"model__C": [0.1, 1, 10],
                     "model__epsilon": [0.01, 0.1, 1]}
                ),
                "gradient_boosting": (
                    GradientBoostingRegressor(),
                    {"model__n_estimators": [50, 100],
                     "model__learning_rate": [0.01, 0.1]}
                ),
                "stacking": (
                    StackingRegressor(
                        estimators=[
                            ("rf", RandomForestRegressor()),
                            ("gb", GradientBoostingRegressor())
                        ],
                        final_estimator=SVR()
                    ),
                    {}  # Hyperparameters for stacking can be added if desired.
                )
            }
            scoring_metric = "r2"

        model_results = {}
        best_model_name = None
        best_model_score = float("-inf")
        best_model_path = None
        best_pipeline = None

        # Use robust cross-validation
        cv_strategy = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)

        # Iterate candidate models for hyperparameter tuning
        for model_name, (model, param_grid) in candidate_models.items():
            pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
            grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=cv_strategy, scoring=scoring_metric)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            try:
                grid_search.fit(X_train, y_train)
                best_estimator = grid_search.best_estimator_
                cv_score = grid_search.best_score_
                y_pred = best_estimator.predict(X_test)

                if is_classification:
                    try:
                        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                    except Exception as ex:
                        logging.warning(f"Error calculating f1_score: {ex}")
                        f1 = 0
                    model_results[model_name] = {
                        "best_params": grid_search.best_params_,
                        "cv_accuracy": round(cv_score, 4),
                        "f1_score": round(f1, 4)
                    }
                else:
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    model_results[model_name] = {
                        "best_params": grid_search.best_params_,
                        "cv_r2": round(cv_score, 4),
                        "mse": round(mse, 4),
                        "r2": round(r2, 4)
                    }

                if cv_score > best_model_score:
                    best_model_score = cv_score
                    best_model_name = model_name
                    best_pipeline = best_estimator
                    best_model_path = os.path.join(MODEL_DIR, f"{model_name}_tuned_model.joblib")
                    joblib.dump(best_pipeline, best_model_path)

            except Exception as e:
                logging.error(f"Error tuning {model_name}: {str(e)}")
                model_results[model_name] = {"error": str(e)}
                continue

        if best_model_name is None:
            return JSONResponse(status_code=500, content={"error": "All hyperparameter tuning attempts failed"})

        response = {
            "best_model_name": best_model_name,
            "best_model_cv_score": round(best_model_score, 4),
            "problem_type": "classification" if is_classification else "regression",
            "evaluation_metrics_all_models": convert_numpy_types(model_results),
            "best_model_hyperparameters": grid_search.best_params_ if best_pipeline else {},
            "best_model_download_link": f"/download-model/{os.path.basename(best_model_path)}"
        }

        # Optionally include plots for further diagnostics
        if include_metrics_plots:
            if is_classification:
                cm = confusion_matrix(y_test, best_pipeline.predict(X_test))
                plt.figure(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                plt.title(f"Confusion Matrix for {best_model_name}")
                buf = BytesIO()
                plt.savefig(buf, format="png")
                plt.close()
                response["confusion_matrix"] = base64.b64encode(buf.getvalue()).decode("utf-8")
            else:
                plt.figure(figsize=(6, 4))
                plt.scatter(y_test, best_pipeline.predict(X_test), alpha=0.7)
                plt.xlabel("Actual")
                plt.ylabel("Predicted")
                plt.title(f"Actual vs Predicted for {best_model_name}")
                buf = BytesIO()
                plt.savefig(buf, format="png")
                plt.close()
                response["predicted_vs_actual"] = base64.b64encode(buf.getvalue()).decode("utf-8")

        return response

    except Exception as e:
        logging.error(f"Error during hyperparameter tuning: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})
    

@app.get("/download-model/{model_name}")
async def download_model(model_name: str):
    """Download a trained model."""
    file_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "Model not found"})
    return FileResponse(file_path, media_type="application/octet-stream", filename=model_name)

@app.post("/test-model/")
async def test_model(model_file: UploadFile = File(...), test_data_file: UploadFile = File(...), target_column: str = Form(...)):
    """Test a trained model on a test dataset."""
    try:
        model_path = os.path.join(TEMP_DIR, "uploaded_model.joblib")
        with open(model_path, "wb") as f:
            f.write(await model_file.read())
        model = joblib.load(model_path)

        test_df = pd.read_csv(test_data_file.file)
        if "Date" in test_df.columns:
            test_df["Date"] = pd.to_datetime(test_df["Date"], errors="coerce")
        if target_column not in test_df.columns:
            return JSONResponse(status_code=400, content={"error": f"Target column '{target_column}' not found"})

        X_test, y_test = test_df.drop(columns=[target_column]), test_df[target_column]
        y_pred = model.predict(X_test)

        is_classification = isinstance(y_test.iloc[0], (str, bool)) or y_test.nunique() <= 20
        if is_classification:
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
        return convert_numpy_types(result)
    except Exception as e:
        logging.error(f"Error testing model: {str(e)}\n{traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, port=8000)