# Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configure the Streamlit page
st.set_page_config(
    page_title="GlassVision",
    page_icon="🔬",
    layout="wide"
)

# Load the dataset only once
@st.cache_data
def load_data():
    return pd.read_csv("glass.csv")

# Load the scaler and models only once
@st.cache_resource
def load_models():
    with open("StandardScaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("SVM_model.pkl", "rb") as f:
        svm_model = pickle.load(f)
    with open("LOG_model.pkl", "rb") as f:
        log_model = pickle.load(f)
    with open("KNN_model.pkl", "rb") as f:
        knn_model = pickle.load(f)
    return (scaler, svm_model, log_model, knn_model)

# Prepare evaluation data and predictions only once
@st.cache_data
def prepare_evaluation_data(_scaler, _svm_model, _log_model, _knn_model, data):
    X = data.drop("Type", axis=1)
    Y = data["Type"]
    X_scaled = _scaler.transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=48)
    SVM_predict = _svm_model.predict(X_test)
    LOG_predict = _log_model.predict(X_test)
    KNN_predict = _knn_model.predict(X_test)
    return (Y_test, SVM_predict, LOG_predict, KNN_predict)

# Load the dataset
data = load_data()

# Load the scaler and models
scaler, svm_model, log_model, knn_model = load_models()

# Define glass type names
glass_types = {
    1: "Building Windows (Float Processed)",
    2: "Building Windows (Non-Float Processed)",
    3: "Vehicle Windows (Float Processed)",
    4: "Vehicle Windows (Non-Float Processed)",
    5: "Containers",
    6: "Tableware",
    7: "Headlamps"
}

# Prepare model evaluation data
Y_test, SVM_predict, LOG_predict, KNN_predict = prepare_evaluation_data(
    scaler, svm_model, log_model, knn_model, data
)

# Create the sidebar title
st.sidebar.title("🔬 GlassVision")

# Add application description
st.sidebar.write("Glass Type Classification using Machine Learning")

# Add a divider
st.sidebar.divider()

# Create navigation menu
page = st.sidebar.radio(
    "Navigate",
    [
        "📊 EDA",
        "⚖️ Model Comparison",
        "🔮 Predictor"
    ]
)

# Create the user input function
def get_user_input():
    # Display the input section
    st.subheader("Enter Glass Composition Values")

    # Let user choose input mode
    input_mode = st.radio(
        "Input Method",
        ["✍️ Manual Input", "🎲 Load Sample"],
        horizontal=True,
        key="input_mode"
    )

    # Initialize default values from dataset means
    defaults = data.drop("Type", axis=1).mean().to_dict()

    # Handle sample loading mode
    if input_mode == "🎲 Load Sample":
        # Explain the sample option
        st.info("Pick a row index from the dataset to auto-fill the values below.")

        # Initialize session state index if not present
        if "sample_idx" not in st.session_state:
            st.session_state["sample_idx"] = 0

        # Sample index picker
        sample_idx = st.number_input(
            "Sample row index",
            min_value=0,
            max_value=len(data) - 1,
            value=st.session_state["sample_idx"],
            step=1,
            key="idx_input"
        )
        # Update session state with chosen index
        st.session_state["sample_idx"] = sample_idx

        # Show the valid row range for reference
        st.caption(f"Valid row range: 0 to {len(data) - 1} ({len(data)} total rows)")

        # Pull the sample row values using the current session state index
        sample_row = data.drop("Type", axis=1).iloc[st.session_state["sample_idx"]]
        defaults = sample_row.to_dict()

        # Show the actual glass type for that sample
        actual_type = int(data.iloc[st.session_state["sample_idx"]]["Type"])
        st.caption(f"Actual Type for row {st.session_state['sample_idx']}: {actual_type} — {glass_types[actual_type]}")

        # Build a live preview of the selected sample row
        preview_df = pd.DataFrame([sample_row])
        # Add the actual type to the preview
        preview_df["Type"] = actual_type
        # Display the live preview table
        st.write("**Selected Row Preview:**")
        st.dataframe(preview_df, use_container_width=True, hide_index=True)

    else:
        # Explain manual input
        st.info("Enter the chemical composition values manually.")

    # Use the sample index (if any) as part of widget keys so fields refresh on change
    refresh_key = st.session_state.get("sample_idx", "manual")

    # Create three input columns
    col1, col2, col3 = st.columns(3)

    # First column
    with col1:
        RI = st.number_input(
            "RI",
            min_value=float(data["RI"].min()),
            max_value=float(data["RI"].max()),
            value=float(defaults["RI"]),
            key=f"RI_{input_mode}_{refresh_key}"
        )
        Na = st.number_input(
            "Na",
            min_value=float(data["Na"].min()),
            max_value=float(data["Na"].max()),
            value=float(defaults["Na"]),
            key=f"Na_{input_mode}_{refresh_key}"
        )
        Mg = st.number_input(
            "Mg",
            min_value=float(data["Mg"].min()),
            max_value=float(data["Mg"].max()),
            value=float(defaults["Mg"]),
            key=f"Mg_{input_mode}_{refresh_key}"
        )

    # Second column
    with col2:
        Al = st.number_input(
            "Al",
            min_value=float(data["Al"].min()),
            max_value=float(data["Al"].max()),
            value=float(defaults["Al"]),
            key=f"Al_{input_mode}_{refresh_key}"
        )
        Si = st.number_input(
            "Si",
            min_value=float(data["Si"].min()),
            max_value=float(data["Si"].max()),
            value=float(defaults["Si"]),
            key=f"Si_{input_mode}_{refresh_key}"
        )
        K = st.number_input(
            "K",
            min_value=float(data["K"].min()),
            max_value=float(data["K"].max()),
            value=float(defaults["K"]),
            key=f"K_{input_mode}_{refresh_key}"
        )

    # Third column
    with col3:
        Ca = st.number_input(
            "Ca",
            min_value=float(data["Ca"].min()),
            max_value=float(data["Ca"].max()),
            value=float(defaults["Ca"]),
            key=f"Ca_{input_mode}_{refresh_key}"
        )
        Ba = st.number_input(
            "Ba",
            min_value=float(data["Ba"].min()),
            max_value=float(data["Ba"].max()),
            value=float(defaults["Ba"]),
            key=f"Ba_{input_mode}_{refresh_key}"
        )
        Fe = st.number_input(
            "Fe",
            min_value=float(data["Fe"].min()),
            max_value=float(data["Fe"].max()),
            value=float(defaults["Fe"]),
            key=f"Fe_{input_mode}_{refresh_key}"
        )

    # Combine input values
    input_data = np.array([RI, Na, Mg, Al, Si, K, Ca, Ba, Fe]).reshape(1, -1)

    # Scale the input
    scaled_input = scaler.transform(input_data)

    # Return scaled input
    return scaled_input

# Function to show predictions from all three models combined into one result
def show_prediction(input_data):
    # Display section title
    st.subheader("Prediction Result")

    # Generate predictions from all three models
    svm_pred = int(svm_model.predict(input_data)[0])
    log_pred = int(log_model.predict(input_data)[0])
    knn_pred = int(knn_model.predict(input_data)[0])

    # Build result table combining all three models
    result = pd.DataFrame({
        "Model": ["SVM", "Logistic Regression", "KNN"],
        "Predicted Type": [svm_pred, log_pred, knn_pred],
        "Glass Category": [
            glass_types[svm_pred],
            glass_types[log_pred],
            glass_types[knn_pred]
        ]
    })

    # Display result table
    st.dataframe(result, use_container_width=True, hide_index=True)

    # Check if all models agree
    if svm_pred == log_pred == knn_pred:
        st.success(f"✅ Predicted Glass Type: {svm_pred} — {glass_types[svm_pred]}")
    else:
        # Use majority vote when models disagree
        votes = pd.Series([svm_pred, log_pred, knn_pred])
        majority_type = int(votes.mode()[0])
        st.warning(
            f"⚠️ Models disagree. Majority vote: Type {majority_type} — {glass_types[majority_type]}"
        )

# Function to handle batch prediction from an uploaded CSV using all three models
def handle_batch_prediction():
    # Display section title
    st.subheader("📂 Batch Prediction from CSV")
    # Explain expected format
    st.write("Upload a CSV file with the following columns: RI, Na, Mg, Al, Si, K, Ca, Ba, Fe")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="batch_upload")

    # Process the uploaded file
    if uploaded_file is not None:
        try:
            # Read the uploaded CSV
            batch_data = pd.read_csv(uploaded_file)
            # Define required columns
            required_cols = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]
            # Check for missing columns
            missing_cols = [c for c in required_cols if c not in batch_data.columns]

            # Show error if columns are missing
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
            else:
                # Extract and scale the feature columns
                X_batch = batch_data[required_cols]
                X_batch_scaled = scaler.transform(X_batch)

                # Generate predictions from all three models
                result = batch_data.copy()
                result["SVM Prediction"] = svm_model.predict(X_batch_scaled)
                result["Logistic Regression Prediction"] = log_model.predict(X_batch_scaled)
                result["KNN Prediction"] = knn_model.predict(X_batch_scaled)

                # Display the results
                st.success(f"Predictions generated for {len(result)} rows.")
                st.dataframe(result, use_container_width=True)

                # Prepare CSV for download
                csv_bytes = result.to_csv(index=False).encode("utf-8")
                # Display download button
                st.download_button(
                    "⬇️ Download Predictions as CSV",
                    data=csv_bytes,
                    file_name="glass_predictions.csv",
                    mime="text/csv",
                    key="batch_download"
                )
        except Exception as e:
            # Display any processing errors
            st.error(f"Error processing file: {e}")

# Display EDA page
if page == "📊 EDA":
    # Display the page title
    st.title("📊 Exploratory Data Analysis")
    # Display the page description
    st.write("Explore the glass composition dataset and understand the distribution of different glass types.")

    # Display dataset overview
    st.subheader("Dataset Overview")
    # Create metric columns
    col1, col2, col3, col4 = st.columns(4)
    # Display total samples
    with col1:
        st.metric("Total Samples", data.shape[0])
    # Display number of features
    with col2:
        st.metric("Input Features", data.shape[1] - 1)
    # Display number of glass types
    with col3:
        st.metric("Glass Types", data["Type"].nunique())
    # Display missing values
    with col4:
        st.metric("Missing Values", data.isnull().sum().sum())

    # Display glass type reference
    st.subheader("Glass Type Reference")
    # Create glass type reference table
    type_reference = pd.DataFrame({
        "Type Number": list(glass_types.keys()),
        "Glass Category": list(glass_types.values())
    })
    # Display glass type reference table
    st.dataframe(type_reference, use_container_width=True, hide_index=True)

    # Display dataset preview
    st.subheader("Dataset Preview")
    # Display the dataset
    st.dataframe(data, use_container_width=True)

    # Display statistical summary
    st.subheader("Statistical Summary")
    # Display descriptive statistics
    st.dataframe(data.describe(), use_container_width=True)

    # Display glass type distribution
    st.subheader("Glass Type Distribution")
    # Count samples by glass type
    type_counts = data["Type"].value_counts().sort_index().reset_index()
    # Rename columns
    type_counts.columns = ["Type", "Count"]
    # Add glass category names
    type_counts["Glass Category"] = type_counts["Type"].map(glass_types)
    # Create the distribution chart
    fig, ax = plt.subplots(figsize=(12, 6))
    # Create the bar chart
    sns.barplot(data=type_counts, x="Glass Category", y="Count", ax=ax)
    # Set chart labels
    ax.set_xlabel("Glass Category")
    ax.set_ylabel("Number of Samples")
    # Rotate labels
    plt.xticks(rotation=30, ha="right")
    # Display chart
    st.pyplot(fig)
    # Close chart
    plt.close(fig)

    # Display feature distribution
    st.subheader("Feature Distribution")
    # Select a feature
    selected_feature = st.selectbox("Select a feature", data.columns[:-1])
    # Create feature distribution chart
    fig, ax = plt.subplots(figsize=(10, 5))
    # Create histogram
    sns.histplot(data=data, x=selected_feature, kde=True, ax=ax)
    # Set chart title
    ax.set_title(f"Distribution of {selected_feature}")
    # Display chart
    st.pyplot(fig)
    # Close chart
    plt.close(fig)

    # Display feature boxplot
    st.subheader("Feature Boxplot")
    # Select feature for boxplot
    selected_boxplot_feature = st.selectbox(
        "Select a feature for boxplot",
        data.columns[:-1],
        key="boxplot_feature"
    )
    # Create boxplot
    fig, ax = plt.subplots(figsize=(10, 5))
    # Display boxplot
    sns.boxplot(data=data, x=selected_boxplot_feature, ax=ax)
    # Set chart title
    ax.set_title(f"Boxplot of {selected_boxplot_feature}")
    # Display chart
    st.pyplot(fig)
    # Close chart
    plt.close(fig)

    # Display feature correlation
    st.subheader("Feature Correlation")
    # Create correlation heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    # Display correlation heatmap
    sns.heatmap(data.corr(), annot=True, annot_kws={"size": 9}, ax=ax)
    # Display heatmap
    st.pyplot(fig)
    # Close heatmap
    plt.close(fig)

    # Display average values by glass type
    st.subheader("Average Feature Values by Glass Type")
    # Calculate average feature values
    feature_means = data.groupby("Type").mean()
    # Add glass type names to the index
    feature_means.index = [
        f"{glass_type}: {glass_types[glass_type]}"
        for glass_type in feature_means.index
    ]
    # Display average feature values
    st.dataframe(feature_means, use_container_width=True)

# Display model comparison page
elif page == "⚖️ Model Comparison":
    # Display page title
    st.title("⚖️ Model Comparison")
    # Display page description
    st.write("Compare the performance of the three trained machine learning models.")

    # Calculate SVM accuracy
    svm_accuracy = accuracy_score(Y_test, SVM_predict)
    # Calculate Logistic Regression accuracy
    log_accuracy = accuracy_score(Y_test, LOG_predict)
    # Calculate KNN accuracy
    knn_accuracy = accuracy_score(Y_test, KNN_predict)

    # Create comparison DataFrame
    comparison_data = pd.DataFrame({
        "Model": ["SVM", "Logistic Regression", "KNN"],
        "Accuracy": [svm_accuracy, log_accuracy, knn_accuracy]
    })
    # Convert accuracy to percentage
    comparison_data["Accuracy (%)"] = (comparison_data["Accuracy"] * 100).round(2)

    # Display model accuracy
    st.subheader("Model Accuracy")
    # Create metric columns
    col1, col2, col3 = st.columns(3)
    # Display SVM accuracy
    with col1:
        st.metric("SVM", f"{svm_accuracy * 100:.2f}%")
    # Display Logistic Regression accuracy
    with col2:
        st.metric("Logistic Regression", f"{log_accuracy * 100:.2f}%")
    # Display KNN accuracy
    with col3:
        st.metric("KNN", f"{knn_accuracy * 100:.2f}%")

    # Display comparison table
    st.subheader("Accuracy Comparison")
    # Display accuracy table
    st.dataframe(
        comparison_data[["Model", "Accuracy (%)"]],
        use_container_width=True,
        hide_index=True
    )

    # Display accuracy visualization
    st.subheader("Accuracy Visualization")
    # Create accuracy chart
    fig, ax = plt.subplots(figsize=(10, 5))
    # Create accuracy bar chart
    sns.barplot(data=comparison_data, x="Model", y="Accuracy (%)", ax=ax)
    # Set y-axis limits
    ax.set_ylim(0, 100)
    # Set chart labels
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Model")
    # Display chart
    st.pyplot(fig)
    # Close chart
    plt.close(fig)

    # Display classification reports
    st.subheader("Detailed Classification Reports")
    # Create model tabs
    tab1, tab2, tab3 = st.tabs(["SVM", "Logistic Regression", "KNN"])
    # Display SVM report
    with tab1:
        svm_report = classification_report(Y_test, SVM_predict, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(svm_report).transpose(), use_container_width=True)
    # Display Logistic Regression report
    with tab2:
        log_report = classification_report(Y_test, LOG_predict, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(log_report).transpose(), use_container_width=True)
    # Display KNN report
    with tab3:
        knn_report = classification_report(Y_test, KNN_predict, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(knn_report).transpose(), use_container_width=True)

    # Display confusion matrices
    st.subheader("Confusion Matrix Comparison")
    # Create confusion matrix tabs
    tab1, tab2, tab3 = st.tabs(["SVM", "Logistic Regression", "KNN"])
    # Get glass type labels
    labels = sorted(data["Type"].unique())
    # Display SVM confusion matrix
    with tab1:
        fig, ax = plt.subplots()
        sns.heatmap(
            confusion_matrix(Y_test, SVM_predict, labels=labels),
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close(fig)
    # Display Logistic Regression confusion matrix
    with tab2:
        fig, ax = plt.subplots()
        sns.heatmap(
            confusion_matrix(Y_test, LOG_predict, labels=labels),
            annot=True,
            fmt="d",
            cmap="Greens",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close(fig)
    # Display KNN confusion matrix
    with tab3:
        fig, ax = plt.subplots()
        sns.heatmap(
            confusion_matrix(Y_test, KNN_predict, labels=labels),
            annot=True,
            fmt="d",
            cmap="Oranges",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close(fig)

    # Find the best model
    best_model = comparison_data.loc[comparison_data["Accuracy"].idxmax(), "Model"]
    # Get the best accuracy
    best_accuracy = comparison_data["Accuracy"].max()
    # Display the best model
    st.success(f"Best performing model: {best_model} with an accuracy of {best_accuracy * 100:.2f}%.")

# Display Predictor page
elif page == "🔮 Predictor":
    # Display page title
    st.title("🔮 Predictor")
    # Display page description
    st.write("Enter glass composition values manually or load a sample row, then predict the glass type.")

    # Get user input
    input_data = get_user_input()

    # Predict glass type
    if st.button("🔮 Predict Glass Type", type="primary"):
        # Show the combined prediction result
        show_prediction(input_data)

    # Display divider before batch section
    st.divider()
    # Show batch prediction section
    handle_batch_prediction()