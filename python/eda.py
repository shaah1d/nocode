# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
import os
from typing import Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import shapiro
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

# Temporary directory to store processed files
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)


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

if __name__ == "__main__":
    uvicorn.run(app, port=8000)