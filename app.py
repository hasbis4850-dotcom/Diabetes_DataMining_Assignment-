# ============================================================
# Q1 - DATA MINING APPLICATION
# Diabetes Prediction Dataset
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Diabetes Data Mining Application",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🩺 Diabetes Data Mining Application")

st.write(
    """
    This interactive application performs data loading, preprocessing,
    exploratory data analysis, data mining and diabetes prediction
    using machine learning techniques.
    """
)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    file_path = "diabetes_prediction_dataset.csv"

    df = pd.read_csv(file_path)

    return df


try:

    df = load_data()

except FileNotFoundError:

    st.error(
        "Dataset not found! Please make sure "
        "'diabetes_prediction_dataset.csv' is inside the project folder."
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Application Menu")

option = st.sidebar.radio(
    "Select an operation:",
    [
        "Dataset Overview",
        "Data Preprocessing",
        "Histogram",
        "Boxplot",
        "Scatterplot",
        "Correlation Heatmap",
        "Data Mining / Model",
        "Prediction"
    ]
)


# ============================================================
# COMMON INFORMATION
# ============================================================

numeric_columns = [
    "age",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level"
]

categorical_columns = [
    "gender",
    "smoking_history"
]

target_column = "diabetes"


# ============================================================
# DATASET OVERVIEW
# ============================================================

if option == "Dataset Overview":

    st.header("1. Dataset Overview")

    st.subheader("Dataset Shape")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric(
            "Duplicate Rows",
            df.duplicated().sum()
        )

    st.subheader("First 10 Records")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.subheader("Dataset Information")

    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum().values,
        "Unique Values": [
            df[column].nunique()
            for column in df.columns
        ]
    })

    st.dataframe(
        info_df,
        use_container_width=True
    )

    st.subheader("Statistical Summary")

    st.dataframe(
        df.describe(include="all").T,
        use_container_width=True
    )

    st.subheader("Diabetes Class Distribution")

    class_counts = df["diabetes"].value_counts()

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar(
        ["No Diabetes", "Diabetes"],
        [
            class_counts.get(0, 0),
            class_counts.get(1, 0)
        ]
    )

    ax.set_title("Diabetes Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Records")

    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# DATA PREPROCESSING
# ============================================================

elif option == "Data Preprocessing":

    st.header("2. Data Loading & Preprocessing")

    st.subheader("Missing Values Before Processing")

    missing_values = df.isnull().sum()

    missing_df = pd.DataFrame({
        "Column": missing_values.index,
        "Missing Values": missing_values.values
    })

    st.dataframe(
        missing_df,
        use_container_width=True
    )

    st.subheader("Duplicate Records")

    duplicate_count = df.duplicated().sum()

    st.write(
        f"Total duplicate records: **{duplicate_count}**"
    )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    processed_df = df.drop_duplicates().copy()

    st.success(
        f"Duplicate handling completed. "
        f"Records after removing duplicates: {len(processed_df)}"
    )

    # --------------------------------------------------------
    # Missing values handling
    # --------------------------------------------------------

    for column in processed_df.columns:

        if processed_df[column].dtype == "object":

            processed_df[column] = processed_df[column].fillna(
                processed_df[column].mode()[0]
                if not processed_df[column].mode().empty
                else "Unknown"
            )

        else:

            processed_df[column] = processed_df[column].fillna(
                processed_df[column].median()
            )

    st.subheader("Missing Values After Processing")

    st.dataframe(
        pd.DataFrame({
            "Column": processed_df.columns,
            "Missing Values": processed_df.isnull().sum().values
        }),
        use_container_width=True
    )

    # --------------------------------------------------------
    # Outlier inspection
    # --------------------------------------------------------

    st.subheader("Outlier Inspection Using IQR Method")

    outlier_results = []

    for column in numeric_columns:

        if column in processed_df.columns:

            Q1 = processed_df[column].quantile(0.25)
            Q3 = processed_df[column].quantile(0.75)

            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            outliers = (
                (processed_df[column] < lower)
                |
                (processed_df[column] > upper)
            ).sum()

            outlier_results.append({
                "Feature": column,
                "Lower Bound": round(lower, 2),
                "Upper Bound": round(upper, 2),
                "Outliers": int(outliers)
            })

    outlier_df = pd.DataFrame(outlier_results)

    st.dataframe(
        outlier_df,
        use_container_width=True
    )

    st.info(
        "Outliers are inspected using the Interquartile Range (IQR) "
        "method. Extreme values are retained because they can represent "
        "genuine patient measurements."
    )

    # --------------------------------------------------------
    # Encoding
    # --------------------------------------------------------

    st.subheader("Categorical Feature Encoding")

    st.write(
        """
        Categorical features such as Gender and Smoking History
        are converted into numerical values using One-Hot Encoding.
        """
    )

    encoded_preview = pd.get_dummies(
        processed_df.drop(columns=[target_column]),
        columns=[
            column
            for column in categorical_columns
            if column in processed_df.columns
        ],
        drop_first=True
    )

    st.dataframe(
        encoded_preview.head(10),
        use_container_width=True
    )

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    st.subheader("Feature Scaling / Normalization")

    scaler = StandardScaler()

    scaling_demo = processed_df[numeric_columns].copy()

    scaled_values = scaler.fit_transform(
        scaling_demo
    )

    scaled_df = pd.DataFrame(
        scaled_values,
        columns=numeric_columns
    )

    st.dataframe(
        scaled_df.head(10),
        use_container_width=True
    )

    st.success(
        "Preprocessing includes duplicate handling, missing-value "
        "handling, outlier inspection, categorical encoding and "
        "feature scaling."
    )


# ============================================================
# HISTOGRAM
# ============================================================

elif option == "Histogram":

    st.header("3. Exploratory Data Analysis — Histogram")

    selected_column = st.selectbox(
        "Select a numerical feature:",
        numeric_columns
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.histplot(
        df[selected_column],
        bins=30,
        kde=True,
        ax=ax
    )

    ax.set_title(
        f"Distribution of {selected_column}"
    )

    ax.set_xlabel(selected_column)
    ax.set_ylabel("Frequency")

    st.pyplot(fig)

    plt.close(fig)

    st.write(
        f"""
        **Observation:** The histogram shows the distribution of
        `{selected_column}` and helps identify its central tendency,
        spread and possible skewness.
        """
    )


# ============================================================
# BOXPLOT
# ============================================================

elif option == "Boxplot":

    st.header("4. Exploratory Data Analysis — Boxplot")

    selected_column = st.selectbox(
        "Select a numerical feature:",
        numeric_columns
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.boxplot(
        x=df[selected_column],
        ax=ax
    )

    ax.set_title(
        f"Boxplot of {selected_column}"
    )

    ax.set_xlabel(selected_column)

    st.pyplot(fig)

    plt.close(fig)

    st.write(
        """
        **Observation:** The boxplot displays the median,
        quartiles and possible outliers of the selected feature.
        """)


# ============================================================
# SCATTERPLOT
# ============================================================

elif option == "Scatterplot":

    st.header("5. Exploratory Data Analysis — Scatterplot")

    col1, col2 = st.columns(2)

    with col1:

        x_feature = st.selectbox(
            "Select X-axis feature:",
            numeric_columns,
            index=0
        )

    with col2:

        y_feature = st.selectbox(
            "Select Y-axis feature:",
            numeric_columns,
            index=2
        )

    fig, ax = plt.subplots(figsize=(9, 6))

    scatter = ax.scatter(
        df[x_feature],
        df[y_feature],
        c=df[target_column],
        alpha=0.4
    )

    ax.set_title(
        f"{x_feature} vs {y_feature}"
    )

    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)

    st.pyplot(fig)

    plt.close(fig)

    st.write(
        """
        **Observation:** The scatterplot helps identify the
        relationship or pattern between two numerical features.
        """)


# ============================================================
# CORRELATION HEATMAP
# ============================================================

elif option == "Correlation Heatmap":

    st.header("6. Exploratory Data Analysis — Correlation Heatmap")

    correlation_df = df.copy()

    # Convert categorical values to numerical codes
    for column in correlation_df.select_dtypes(
        include="object"
    ).columns:

        correlation_df[column] = pd.factorize(
            correlation_df[column]
        )[0]

    correlation_matrix = correlation_df.corr(
        numeric_only=True
    )

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax
    )

    ax.set_title(
        "Correlation Heatmap"
    )

    st.pyplot(fig)

    plt.close(fig)

    st.write(
        """
        **Observation:** The correlation heatmap shows the strength
        and direction of relationships between numerical variables.
        Values closer to +1 indicate strong positive relationships,
        while values closer to -1 indicate strong negative relationships.
        """)


# ============================================================
# DATA MINING / MACHINE LEARNING
# ============================================================

elif option == "Data Mining / Model":

    st.header("7. Data Mining / Analysis")

    st.write(
        """
        Since the target variable `diabetes` contains two classes
        (0 = No Diabetes and 1 = Diabetes), classification is an
        appropriate data mining technique.
        """
    )

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    model_df = df.drop_duplicates().copy()

    X = model_df.drop(
        columns=[target_column]
    )

    y = model_df[target_column]

    numeric_features = [
        column
        for column in numeric_columns
        if column in X.columns
    ]

    categorical_features = [
        column
        for column in categorical_columns
        if column in X.columns
    ]

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                numeric_features
            ),
            (
                "cat",
                categorical_transformer,
                categorical_features
            )
        ]
    )

    # --------------------------------------------------------
    # Train-test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=1000
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                random_state=42,
                max_depth=8
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
    }

    results = []

    trained_models = {}

    for model_name, model in models.items():

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    model
                )
            ]
        )

        pipeline.fit(
            X_train,
            y_train
        )

        predictions = pipeline.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        results.append({
            "Model": model_name,
            "Accuracy": round(
                accuracy * 100,
                2
            )
        })

        trained_models[model_name] = (
            pipeline,
            predictions
        )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    st.subheader("Model Comparison")

    st.dataframe(
        results_df,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Accuracy graph
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.bar(
        results_df["Model"],
        results_df["Accuracy"]
    )

    ax.set_title(
        "Machine Learning Model Accuracy"
    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy (%)")

    ax.set_ylim(0, 100)

    plt.xticks(
        rotation=15
    )

    st.pyplot(fig)

    plt.close(fig)

    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    best_model_name = results_df.loc[
        results_df["Accuracy"].idxmax(),
        "Model"
    ]

    best_accuracy = results_df["Accuracy"].max()

    st.success(
        f"Best Model: {best_model_name} "
        f"with accuracy of {best_accuracy}%"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    best_pipeline = trained_models[
        best_model_name
    ][0]

    best_predictions = trained_models[
        best_model_name
    ][1]

    st.subheader(
        f"Classification Report — {best_model_name}"
    )

    report = classification_report(
        y_test,
        best_predictions,
        output_dict=True
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    st.dataframe(
        report_df,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        best_predictions
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "Actual"
    )

    ax.set_title(
        "Confusion Matrix"
    )

    st.pyplot(fig)

    plt.close(fig)

    st.session_state["best_model"] = best_pipeline


# ============================================================
# PREDICTION
# ============================================================

elif option == "Prediction":

    st.header("8. Diabetes Prediction")

    st.write(
        """
        Enter the patient's information below.
        The trained Random Forest model will predict whether
        the patient is likely to have diabetes.
        """
    )

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    model_df = df.drop_duplicates().copy()

    X = model_df.drop(
        columns=[target_column]
    )

    y = model_df[target_column]

    numeric_features = [
        column
        for column in numeric_columns
        if column in X.columns
    ]

    categorical_features = [
        column
        for column in categorical_columns
        if column in X.columns
    ]

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                numeric_features
            ),
            (
                "cat",
                categorical_transformer,
                categorical_features
            )
        ]
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10
    )

    prediction_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    prediction_pipeline.fit(
        X,
        y
    )

    # --------------------------------------------------------
    # User Input
    # --------------------------------------------------------

    st.subheader(
        "Enter Patient Information"
    )

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Female", "Male", "Other"]
        )

        age = st.number_input(
            "Age",
            min_value=1.0,
            max_value=120.0,
            value=40.0,
            step=1.0
        )

        hypertension = st.selectbox(
            "Hypertension",
            [0, 1],
            format_func=lambda x:
            "No" if x == 0 else "Yes"
        )

        heart_disease = st.selectbox(
            "Heart Disease",
            [0, 1],
            format_func=lambda x:
            "No" if x == 0 else "Yes"
        )

    with col2:

        smoking_history = st.selectbox(
            "Smoking History",
            [
                "never",
                "No Info",
                "current",
                "former",
                "ever",
                "not current"
            ]
        )

        bmi = st.number_input(
            "BMI",
            min_value=5.0,
            max_value=80.0,
            value=25.0,
            step=0.1
        )

        hba1c = st.number_input(
            "HbA1c Level",
            min_value=3.0,
            max_value=15.0,
            value=5.5,
            step=0.1
        )

        glucose = st.number_input(
            "Blood Glucose Level",
            min_value=50.0,
            max_value=500.0,
            value=120.0,
            step=1.0
        )

    # --------------------------------------------------------
    # Prediction button
    # --------------------------------------------------------

    if st.button(
        "🔍 Predict Diabetes",
        use_container_width=True
    ):

        input_data = pd.DataFrame({
            "gender": [gender],
            "age": [age],
            "hypertension": [hypertension],
            "heart_disease": [heart_disease],
            "smoking_history": [smoking_history],
            "bmi": [bmi],
            "HbA1c_level": [hba1c],
            "blood_glucose_level": [glucose]
        })

        prediction = prediction_pipeline.predict(
            input_data
        )[0]

        probability = prediction_pipeline.predict_proba(
            input_data
        )[0]

        diabetes_probability = probability[1] * 100

        st.subheader(
            "Prediction Result"
        )

        if prediction == 1:

            st.error(
                "⚠️ Prediction: Diabetes"
            )

            st.write(
                f"Estimated diabetes probability: "
                f"**{diabetes_probability:.2f}%**"
            )

        else:

            st.success(
                "✅ Prediction: No Diabetes"
            )

            st.write(
                f"Estimated diabetes probability: "
                f"**{diabetes_probability:.2f}%**"
            )

        st.subheader(
            "Input Values"
        )

        st.dataframe(
            input_data,
            use_container_width=True
        )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.markdown("---")

st.sidebar.info(
)