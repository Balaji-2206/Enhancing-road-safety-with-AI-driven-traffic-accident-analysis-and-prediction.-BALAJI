import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    roc_auc_score,
    f1_score,
)

from io import BytesIO

st.set_page_config(
    page_title="🚦 Traffic Accident Analysis & Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

def plot_confusion_matrix(cm, title="Confusion Matrix"):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(plt)
    plt.clf()

def plot_roc_curve(y_test, y_probs, model_name="Model"):
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = roc_auc_score(y_test, y_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="blue", label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0,1], [0,1], color="red", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    st.pyplot(plt)
    plt.clf()

def load_and_preprocess_data(uploaded_file):
    df = pd.read_excel(uploaded_file)
    drop_cols = []
    if "Accident_ID" in df.columns:
        drop_cols.append("Accident_ID")
    if "Time" in df.columns:
        drop_cols.append("Time")
    if drop_cols:
        df.drop(drop_cols, axis=1, inplace=True)

    if "Date" in df.columns:
        df["Year"] = pd.to_datetime(df["Date"]).dt.year
        df.drop("Date", axis=1, inplace=True)

    categorical_cols = [
        "Weather_Condition",
        "Road_Type",
        "Light_Condition",
        "Vehicle_Type_Involved",
        "Primary_Cause",
        "Accident_Severity",
    ]

    label_encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    return df, label_encoders

def main():
    st.title("🚦 Enhancing Road Safety with AI-Driven Traffic Accident Analysis and Prediction")
    st.markdown(
        """
        Upload your traffic accident dataset (Excel) and analyze accident severity with multiple AI models.
        Compare baseline and advanced classifiers using various metrics and visualize forecasts.
        """
    )

    menu = st.sidebar.radio(
        "Navigation",
        [
            "1. Data Upload & Preview",
            "2. Exploratory Data Analysis",
            "3. Model Training & Evaluation",
            "4. Forecasting",
        ],
    )

    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'label_encoders' not in st.session_state:
        st.session_state.label_encoders = None
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}

    if menu == "1. Data Upload & Preview":
        uploaded_file = st.file_uploader("📂 Upload Traffic Accidents Dataset (Excel .xlsx)", type=["xlsx"])
        if uploaded_file is not None:
            df, label_encoders = load_and_preprocess_data(uploaded_file)
            st.session_state.df = df
            st.session_state.label_encoders = label_encoders
            st.success(f"✅ Dataset '{uploaded_file.name}' loaded and preprocessed successfully!")
            with st.expander("🧐 Preview Raw Data", expanded=True):
                st.dataframe(df.head())
        else:
            st.info("📥 Please upload an Excel file (.xlsx) to start.")

    if menu == "2. Exploratory Data Analysis":
        if st.session_state.df is not None:
            df = st.session_state.df
            # Sidebar filter for years if available
            if "Year" in df.columns:
                years = sorted(df["Year"].unique())
                selected_years = st.sidebar.multiselect("Filter: Select Year(s):", years, default=years)
                df = df[df["Year"].isin(selected_years)]

            st.subheader("Accident Severity Distribution")
            plt.figure(figsize=(7, 5))
            sns.countplot(data=df, x="Accident_Severity", palette="coolwarm")
            plt.xlabel("Accident Severity")
            plt.ylabel("Count")
            plt.title("Distribution of Accident Severity")
            st.pyplot(plt)
            plt.clf()

            st.subheader("Correlation Heatmap")
            plt.figure(figsize=(12, 8))
            sns.heatmap(df.corr(), annot=True, cmap="viridis", fmt=".2f", square=True)
            plt.title("Correlation between Variables")
            st.pyplot(plt)
            plt.clf()
        else:
            st.warning("Please upload and preprocess data first in 'Data Upload & Preview'.")

    if menu == "3. Model Training & Evaluation":
        if st.session_state.df is not None:
            df = st.session_state.df
            X = df.drop("Accident_Severity", axis=1, errors="ignore")
            y = df["Accident_Severity"]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            st.subheader("Select Models to Train and Evaluate")
            model_options = st.multiselect(
                "Choose models:",
                [
                    "Logistic Regression (Baseline)",
                    "Random Forest Classifier",
                    "Gradient Boosting Classifier",
                ],
                default=["Logistic Regression (Baseline)", "Random Forest Classifier"],
            )

            trained_models = {}
            metrics_summary = []

            for model_name in model_options:
                st.markdown(f"### ▶️ Training: **{model_name}**")
                if model_name == "Logistic Regression (Baseline)":
                    model = LogisticRegression(max_iter=500, random_state=42)
                elif model_name == "Random Forest Classifier":
                    model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
                elif model_name == "Gradient Boosting Classifier":
                    model = GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42)
                else:
                    continue

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test)
                    if y_prob.shape[1] == 2:  # binary
                        y_pred_prob_pos = y_prob[:, 1]
                    else:
                        y_pred_prob_pos = None
                else:
                    y_pred_prob_pos = None

                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="macro")
                cm = confusion_matrix(y_test, y_pred)

                metrics_summary.append(
                    {
                        "Model": model_name,
                        "Accuracy": accuracy,
                        "F1 Score": f1,
                    }
                )
                trained_models[model_name] = model

                st.write(f"Accuracy: **{accuracy:.3f}**")
                st.write(f"Macro F1 Score: **{f1:.3f}**")

                st.text("Classification Report:")
                st.text(classification_report(y_test, y_pred))

                st.subheader(f"Confusion Matrix - {model_name}")
                plot_confusion_matrix(cm, title=f"Confusion Matrix - {model_name}")

                if y_pred_prob_pos is not None and len(np.unique(y_test)) == 2:
                    st.subheader(f"ROC Curve - {model_name}")
                    plot_roc_curve(y_test, y_pred_prob_pos, model_name=model_name)
                else:
                    st.info(
                        "ROC curve is shown only for binary classification with probability output."
                    )

            st.subheader("Summary of Metrics")
            metrics_df = pd.DataFrame(metrics_summary).set_index("Model")
            st.dataframe(metrics_df.style.background_gradient(cmap="viridis"))

            # Save trained models for download
            st.subheader("Download Trained Models")
            for model_name, model in trained_models.items():
                model_buffer = BytesIO()
                joblib.dump(model, model_buffer)
                model_buffer.seek(0)
                safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
                st.download_button(
                    label=f"📥 Download {model_name} (.pkl)",
                    data=model_buffer,
                    file_name=f"{safe_name}.pkl",
                    mime="application/octet-stream",
                )
            st.session_state.trained_models = trained_models

        else:
            st.warning("Please upload and preprocess data first in 'Data Upload & Preview'.")

    if menu == "4. Forecasting":
        if st.session_state.df is not None:
            df = st.session_state.df
            if "Year" in df.columns:
                st.subheader("Traffic Accident Count Forecasting (Linear Regression)")

                accident_counts = df.groupby("Year").size().reset_index(name="Accident_Count")
                lin_reg = LinearRegression()
                lin_reg.fit(accident_counts[["Year"]], accident_counts["Accident_Count"])

                future_years = np.array(range(df["Year"].max() + 1, df["Year"].max() + 6)).reshape(-1, 1)
                future_predictions = lin_reg.predict(future_years)
                future_predictions = np.maximum(future_predictions, 0)  # no negative values

                plt.figure(figsize=(10, 6))
                plt.plot(accident_counts["Year"], accident_counts["Accident_Count"], marker="o", label="Historical")
                plt.plot(future_years, future_predictions, marker="x", linestyle="--", color="red", label="Forecast")
                plt.title("Forecast of Traffic Accidents for Next 5 Years")
                plt.xlabel("Year")
                plt.ylabel("Number of Accidents")
                plt.legend()
                plt.grid(True)
                st.pyplot(plt)
                plt.clf()

                forecast_df = pd.DataFrame({
                    "Year": future_years.flatten(),
                    "Predicted_Accidents": future_predictions.astype(int)
                })
                st.table(forecast_df)

                # Download forecasting model
                linreg_buffer = BytesIO()
                joblib.dump(lin_reg, linreg_buffer)
                linreg_buffer.seek(0)
                st.download_button(
                    label="📥 Download Accident Forecast Model (.pkl)",
                    data=linreg_buffer,
                    file_name="accident_forecast_model.pkl",
                    mime="application/octet-stream",
                )
            else:
                st.info("Year column not found in dataset, cannot do forecasting.")
        else:
            st.warning("Please upload and preprocess data first in 'Data Upload & Preview'.")

if __name__ == "__main__":
    main()
