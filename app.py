import streamlit as st
import pandas as pd
import numpy as np
from pycaret.classification import setup as clf_setup, compare_models as clf_compare, save_model as clf_save, plot_model as clf_plot_model
from pycaret.regression import setup as reg_setup, compare_models as reg_compare, save_model as reg_save, plot_model as reg_plot_model, pull
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import joblib
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# Initialize dataframe and session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None

df = None
if os.path.exists('./dataset.csv'):
    try:
        df = pd.read_csv('dataset.csv', index_col=None)
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")

with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("Neural Nexus No Code Model Maker")
    choice = st.radio("Navigation", ["Upload", "Profiling", "Modelling", "Visuals", "Download"])

def auto_preprocess(df):
    df_clean = df.copy()
    original_columns = set(df_clean.columns)

    # 1. Handle Missing Values
    for col in df_clean.columns:
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            df_clean[col] = df_clean[col].fillna(df_clean[col].median()).astype(float)
        else:
            mode_val = df_clean[col].mode()[0]
            df_clean[col] = df_clean[col].fillna(mode_val)

    # 2. Convert Text Features
    text_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
    binary_cols = []
    for col in text_cols:
        unique_vals = df_clean[col].nunique()
        if unique_vals == 2:
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col]).astype(int)
            binary_cols.append(col)
        else:
            freq = df_clean[col].value_counts(normalize=True)
            df_clean[col] = df_clean[col].map(freq)
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype(float)

    # 3. Smart Scaling
    numeric_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
    non_binary_numeric = [col for col in numeric_cols if col not in binary_cols]
    if non_binary_numeric:
        scaler = StandardScaler()
        df_clean[non_binary_numeric] = scaler.fit_transform(df_clean[non_binary_numeric])
        df_clean[non_binary_numeric] = df_clean[non_binary_numeric].astype(float)

    # 4. Final type conversion
    df_clean = df_clean.convert_dtypes()
    missing_cols = original_columns - set(df_clean.columns)
    if missing_cols:
        raise ValueError(f"Missing columns after preprocessing: {missing_cols}")
    return df_clean

def determine_problem_type(target_series):
    if pd.api.types.is_numeric_dtype(target_series):
        unique_values = target_series.nunique()
        if unique_values / len(target_series) < 0.05:
            return "classification"
        return "regression"
    return "classification"

if choice == "Upload":
    st.title("Data Upload")
    file = st.file_uploader("Upload Dataset (CSV)")
    if file:
        try:
            df = pd.read_csv(file)
            df.to_csv('dataset.csv', index=False)
            st.success("Dataset uploaded!")
            st.write("Original columns:", df.columns.tolist())
            st.dataframe(df.head(3))
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

if choice == "Profiling":
    st.title("Exploratory Analysis")
    if df is not None:
        try:
            profile = ProfileReport(df)
            st_profile_report(profile)
        except Exception as e:
            st.error(f"Profiling failed: {str(e)}")
    else:
        st.warning("Upload dataset first!")

if choice == "Modelling":
    if df is not None:
        try:
            with st.spinner("Preprocessing data..."):
                df_clean = auto_preprocess(df)
                st.write("Processed Data Types:")
                st.write(df_clean.dtypes)
                chosen_target = st.selectbox("Select Target Column", df_clean.columns)
                if chosen_target not in df_clean.columns:
                    st.error("Selected target column not found in processed data")
                    st.stop()
                problem_type = determine_problem_type(df_clean[chosen_target])
                st.success(f"Detected Problem Type: {problem_type.upper()}")
                if st.button("Start Training"):
                    with st.spinner("Configuring Experiment..."):
                        setup_config = {
                            'data': df_clean,
                            'target': chosen_target,
                            'session_id': 42,
                            'verbose': False,
                            'categorical_imputation': 'constant',
                            'numeric_imputation': 'mean',
                            'normalize': False,
                            'fold_strategy': 'stratifiedkfold' if problem_type == 'classification' else 'kfold'
                        }
                        try:
                            if problem_type == "regression":
                                exp = reg_setup(**setup_config)
                                model_list = ['lr', 'dt', 'rf', 'svm', 'knn']
                                best = reg_compare(include=model_list)
                                reg_save(best, 'best_model')
                            else:
                                exp = clf_setup(**setup_config)
                                model_list = ['lr', 'dt', 'rf', 'svm']
                                best = clf_compare(include=model_list)
                                clf_save(best, 'best_model')
                            results = pull()
                            st.session_state.results = results
                            st.session_state.problem_type = problem_type
                            st.write("Model Comparison Results:")
                            st.dataframe(results)
                            st.success("Training Complete!")
                        except Exception as e:
                            st.error(f"Model training failed: {str(e)}")
        except Exception as e:
            st.error(f"Processing error: {str(e)}")
    else:
        st.warning("Upload dataset first!")

if choice == "Visuals":
    st.title("Model Visualizations")
    if not os.path.exists('best_model.pkl') or st.session_state.results is None:
        st.warning("No trained models available. Please train models first.")
    else:
        try:
            model = joblib.load('best_model.pkl')
            results = st.session_state.results
            st.subheader("Trained Models Performance")

            # Create three columns for layout
            cols = st.columns(3)


            # Safe metric formatting function
            def safe_format(value):
                try:
                    return f"{float(value):.4f}" if pd.notnull(value) else "N/A"
                except (ValueError, TypeError):
                    return "N/A"


            # Iterate through results using index and row
            for idx, (model_name, row) in enumerate(results.iterrows()):
                with cols[idx % 3]:
                    with st.expander(f"{model_name}", expanded=False):
                        # Prepare metrics based on problem type
                        if st.session_state.problem_type == "regression":
                            metrics = [
                                ("MAE", row.get('MAE')),
                                ("R²", row.get('R2')),
                                ("RMSE", row.get('RMSE'))
                            ]
                            plot_options = {
                                "Residuals Plot": "residuals",
                                "Prediction Error Plot": "error",
                                "Feature Importance": "feature"
                            }
                        else:
                            metrics = [
                                ("Accuracy", row.get('Accuracy')),
                                ("AUC", row.get('AUC')),
                                ("F1", row.get('F1'))
                            ]
                            plot_options = {
                                "Confusion Matrix": "confusion_matrix",
                                "Feature Importance": "feature",
                                "ROC Curve": "auc"
                            }

                        # Display metrics
                        st.write("**Performance Metrics**")
                        for metric_name, metric_value in metrics:
                            st.write(f"{metric_name}: {safe_format(metric_value)}")

                        # Plot selection
                        selected_plot = st.selectbox(
                            "Select visualization",
                            options=list(plot_options.keys()),
                            key=f"plot_{idx}"
                        )

                        # Generate plot with error handling
                        try:
                            with st.spinner("Generating plot..."):
                                plot_func = reg_plot_model if st.session_state.problem_type == "regression" else clf_plot_model
                                plot_func(
                                    model,
                                    plot=plot_options[selected_plot],
                                    display_format='streamlit'
                                )
                        except Exception as e:
                            st.error(f"Could not generate {selected_plot} plot: {str(e)}")

        except Exception as e:
            st.error(f"Error in visualization: {str(e)}")


if choice == "Download":
    if os.path.exists('best_model.pkl'):
        with open('best_model.pkl', 'rb') as f:
            st.download_button("Download Model", f, "best_model.pkl")
    else:
        st.warning("No trained model available")