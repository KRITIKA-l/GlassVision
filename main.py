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


# Load the glass dataset
data = pd.read_csv("glass.csv")


# Map glass type numbers to their names
glass_types = {
    1: "Building Windows (Float Processed)",
    2: "Building Windows (Non-Float Processed)",
    3: "Vehicle Windows (Float Processed)",
    4: "Vehicle Windows (Non-Float Processed)",
    5: "Containers",
    6: "Tableware",
    7: "Headlamps"
}


# Load the saved scaler
with open("StandardScaler.pkl", "rb") as f:
    scaler = pickle.load(f)


# Load the saved SVM model
with open("SVM_model.pkl", "rb") as f:
    svm_model = pickle.load(f)


# Load the saved Logistic Regression model
with open("LOG_model.pkl", "rb") as f:
    log_model = pickle.load(f)


# Load the saved KNN model
with open("KNN_model.pkl", "rb") as f:
    knn_model = pickle.load(f)


# Separate features and target
X = data.drop(
    "Type",
    axis=1
)

Y = data["Type"]


# Scale the dataset using the saved scaler
X_scaled = scaler.transform(X)


# Split the data to create a test set for model evaluation
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled,
    Y,
    test_size=0.3,
    random_state=48
)


# Generate predictions using the saved models
SVM_predict = svm_model.predict(
    X_test
)

LOG_predict = log_model.predict(
    X_test
)

KNN_predict = knn_model.predict(
    X_test
)


# Create the sidebar title
st.sidebar.title(
    "🔬 GlassVision"
)


# Add a short application description
st.sidebar.write(
    "Glass Type Classification using Machine Learning"
)


# Add a divider in the sidebar
st.sidebar.divider()


# Create the navigation menu
page = st.sidebar.radio(
    "Navigate",
    [
        "📊 EDA",
        "⚖️ Model Comparison",
        "⚙️ SVM",
        "📈 Logistic Regression",
        "🔍 KNN"
    ]
)


# Create a function to collect user input
def get_user_input():

    # Display the input section title
    st.subheader(
        "Enter Glass Composition Values"
    )


    # Explain the input options
    st.info(
        "Enter the chemical composition values manually "
        "or load a random sample from the dataset."
    )


    # Load a random sample when the button is clicked
    if st.button(
        "🎲 Load Sample"
    ):

        # Get the glass types available in the dataset
        available_types = data["Type"].unique()


        # Randomly select one glass type
        selected_type = np.random.choice(
            available_types
        )


        # Select one random sample from the selected type
        sample = data[
            data["Type"] == selected_type
        ].sample(
            1
        ).iloc[0]


        # Store the sample feature values
        st.session_state["RI"] = float(
            sample["RI"]
        )

        st.session_state["Na"] = float(
            sample["Na"]
        )

        st.session_state["Mg"] = float(
            sample["Mg"]
        )

        st.session_state["Al"] = float(
            sample["Al"]
        )

        st.session_state["Si"] = float(
            sample["Si"]
        )

        st.session_state["K"] = float(
            sample["K"]
        )

        st.session_state["Ca"] = float(
            sample["Ca"]
        )

        st.session_state["Ba"] = float(
            sample["Ba"]
        )

        st.session_state["Fe"] = float(
            sample["Fe"]
        )


        # Store the actual glass type
        st.session_state["sample_type"] = int(
            selected_type
        )


        # Rerun the app to display the loaded values
        st.rerun()


    # Display the actual type of the loaded sample
    if "sample_type" in st.session_state:

        # Get the actual type
        actual_type = st.session_state[
            "sample_type"
        ]


        # Display the actual type and category
        st.info(
            f"Loaded Sample | "
            f"Actual Type: {actual_type} - "
            f"{glass_types[actual_type]}"
        )


    # Create three columns for input fields
    col1, col2, col3 = st.columns(
        3
    )


    # Add the first group of input fields
    with col1:

        # Input for refractive index
        RI = st.number_input(
            "RI",
            min_value=float(
                data["RI"].min()
            ),
            max_value=float(
                data["RI"].max()
            ),
            value=st.session_state.get(
                "RI",
                float(
                    data["RI"].mean()
                )
            ),
            key="RI_input"
        )


        # Input for sodium
        Na = st.number_input(
            "Na",
            min_value=float(
                data["Na"].min()
            ),
            max_value=float(
                data["Na"].max()
            ),
            value=st.session_state.get(
                "Na",
                float(
                    data["Na"].mean()
                )
            ),
            key="Na_input"
        )


        # Input for magnesium
        Mg = st.number_input(
            "Mg",
            min_value=float(
                data["Mg"].min()
            ),
            max_value=float(
                data["Mg"].max()
            ),
            value=st.session_state.get(
                "Mg",
                float(
                    data["Mg"].mean()
                )
            ),
            key="Mg_input"
        )


    # Add the second group of input fields
    with col2:

        # Input for aluminium
        Al = st.number_input(
            "Al",
            min_value=float(
                data["Al"].min()
            ),
            max_value=float(
                data["Al"].max()
            ),
            value=st.session_state.get(
                "Al",
                float(
                    data["Al"].mean()
                )
            ),
            key="Al_input"
        )


        # Input for silicon
        Si = st.number_input(
            "Si",
            min_value=float(
                data["Si"].min()
            ),
            max_value=float(
                data["Si"].max()
            ),
            value=st.session_state.get(
                "Si",
                float(
                    data["Si"].mean()
                )
            ),
            key="Si_input"
        )


        # Input for potassium
        K = st.number_input(
            "K",
            min_value=float(
                data["K"].min()
            ),
            max_value=float(
                data["K"].max()
            ),
            value=st.session_state.get(
                "K",
                float(
                    data["K"].mean()
                )
            ),
            key="K_input"
        )


    # Add the third group of input fields
    with col3:

        # Input for calcium
        Ca = st.number_input(
            "Ca",
            min_value=float(
                data["Ca"].min()
            ),
            max_value=float(
                data["Ca"].max()
            ),
            value=st.session_state.get(
                "Ca",
                float(
                    data["Ca"].mean()
                )
            ),
            key="Ca_input"
        )


        # Input for barium
        Ba = st.number_input(
            "Ba",
            min_value=float(
                data["Ba"].min()
            ),
            max_value=float(
                data["Ba"].max()
            ),
            value=st.session_state.get(
                "Ba",
                float(
                    data["Ba"].mean()
                )
            ),
            key="Ba_input"
        )


        # Input for iron
        Fe = st.number_input(
            "Fe",
            min_value=float(
                data["Fe"].min()
            ),
            max_value=float(
                data["Fe"].max()
            ),
            value=st.session_state.get(
                "Fe",
                float(
                    data["Fe"].mean()
                )
            ),
            key="Fe_input"
        )


    # Combine all input values into one array
    input_data = np.array(
        [
            RI,
            Na,
            Mg,
            Al,
            Si,
            K,
            Ca,
            Ba,
            Fe
        ]
    ).reshape(
        1,
        -1
    )


    # Scale the user input
    scaled_input = scaler.transform(
        input_data
    )


    # Return the scaled input
    return scaled_input


# Display the EDA page
if page == "📊 EDA":

    # Display the page title
    st.title(
        "📊 Exploratory Data Analysis"
    )


    # Display the page description
    st.write(
        "Explore the glass composition dataset and "
        "understand the distribution of different glass types."
    )


    # Display dataset overview
    st.subheader(
        "Dataset Overview"
    )


    # Create metric columns
    col1, col2, col3, col4 = st.columns(
        4
    )


    # Display total number of samples
    with col1:

        st.metric(
            "Total Samples",
            data.shape[0]
        )


    # Display number of input features
    with col2:

        st.metric(
            "Input Features",
            data.shape[1] - 1
        )


    # Display number of glass types
    with col3:

        st.metric(
            "Glass Types",
            data["Type"].nunique()
        )


    # Display total missing values
    with col4:

        st.metric(
            "Missing Values",
            data.isnull().sum().sum()
        )


    # Display glass type reference
    st.subheader(
        "Glass Type Reference"
    )


    # Create a reference table for glass types
    type_reference = pd.DataFrame(
        {
            "Type Number": list(
                glass_types.keys()
            ),
            "Glass Category": list(
                glass_types.values()
            )
        }
    )


    # Display the reference table
    st.dataframe(
        type_reference,
        use_container_width=True,
        hide_index=True
    )


    # Display the dataset
    st.subheader(
        "Dataset Preview"
    )


    # Show the dataset in a table
    st.dataframe(
        data,
        use_container_width=True
    )


    # Display statistical summary
    st.subheader(
        "Statistical Summary"
    )


    # Show descriptive statistics
    st.dataframe(
        data.describe(),
        use_container_width=True
    )


    # Display glass type distribution
    st.subheader(
        "Glass Type Distribution"
    )


    # Count samples for each glass type
    type_counts = (
        data["Type"]
        .value_counts()
        .sort_index()
        .reset_index()
    )


    # Rename the columns
    type_counts.columns = [
        "Type",
        "Count"
    ]


    # Add glass category names
    type_counts["Glass Category"] = (
        type_counts["Type"].map(
            glass_types
        )
    )


    # Create the distribution chart
    fig, ax = plt.subplots(
        figsize=(12, 6)
    )


    # Create a bar chart
    sns.barplot(
        data=type_counts,
        x="Glass Category",
        y="Count",
        ax=ax
    )


    # Set chart labels
    ax.set_xlabel(
        "Glass Category"
    )

    ax.set_ylabel(
        "Number of Samples"
    )


    # Rotate category labels
    plt.xticks(
        rotation=30,
        ha="right"
    )


    # Display the chart
    st.pyplot(
        fig
    )


    # Close the chart
    plt.close(
        fig
    )


    # Display feature distribution
    st.subheader(
        "Feature Distribution"
    )


    # Allow the user to select a feature
    selected_feature = st.selectbox(
        "Select a feature",
        data.columns[:-1]
    )


    # Create the distribution plot
    fig, ax = plt.subplots(
        figsize=(10, 5)
    )


    # Create a histogram with KDE
    sns.histplot(
        data=data,
        x=selected_feature,
        kde=True,
        ax=ax
    )


    # Set the chart title
    ax.set_title(
        f"Distribution of {selected_feature}"
    )


    # Display the chart
    st.pyplot(
        fig
    )


    # Close the chart
    plt.close(
        fig
    )


    # Display feature boxplot
    st.subheader(
        "Feature Boxplot"
    )


    # Allow the user to select a feature
    selected_boxplot_feature = st.selectbox(
        "Select a feature for boxplot",
        data.columns[:-1],
        key="boxplot_feature"
    )


    # Create the boxplot
    fig, ax = plt.subplots(
        figsize=(10, 5)
    )


    # Display the selected feature's boxplot
    sns.boxplot(
        data=data,
        x=selected_boxplot_feature,
        ax=ax
    )


    # Set the chart title
    ax.set_title(
        f"Boxplot of {selected_boxplot_feature}"
    )


    # Display the chart
    st.pyplot(
        fig
    )


    # Close the chart
    plt.close(
        fig
    )


    # Display feature correlation
    st.subheader(
        "Feature Correlation"
    )


    # Create the correlation heatmap
    fig, ax = plt.subplots(
        figsize=(12, 6)
    )


    # Display correlations between features
    sns.heatmap(
        data.corr(),
        annot=True,
        annot_kws={
            "size": 9
        },
        ax=ax
    )


    # Display the heatmap
    st.pyplot(
        fig
    )


    # Close the chart
    plt.close(
        fig
    )


    # Display average feature values by glass type
    st.subheader(
        "Average Feature Values by Glass Type"
    )


    # Calculate the average values for each type
    feature_means = (
        data
        .groupby(
            "Type"
        )
        .mean()
    )


    # Add glass type names to the index
    feature_means.index = [
        f"{glass_type}: "
        f"{glass_types[glass_type]}"
        for glass_type in feature_means.index
    ]


    # Display the averages
    st.dataframe(
        feature_means,
        use_container_width=True
    )


# Display the model comparison page
elif page == "⚖️ Model Comparison":

    # Display the page title
    st.title(
        "⚖️ Model Comparison"
    )


    # Display the page description
    st.write(
        "Compare the performance of the three trained "
        "machine learning models."
    )


    # Calculate SVM accuracy
    svm_accuracy = accuracy_score(
        Y_test,
        SVM_predict
    )


    # Calculate Logistic Regression accuracy
    log_accuracy = accuracy_score(
        Y_test,
        LOG_predict
    )


    # Calculate KNN accuracy
    knn_accuracy = accuracy_score(
        Y_test,
        KNN_predict
    )


    # Create a comparison DataFrame
    comparison_data = pd.DataFrame(
        {
            "Model": [
                "SVM",
                "Logistic Regression",
                "KNN"
            ],
            "Accuracy": [
                svm_accuracy,
                log_accuracy,
                knn_accuracy
            ]
        }
    )


    # Convert accuracy into percentages
    comparison_data["Accuracy (%)"] = (
        comparison_data["Accuracy"] * 100
    ).round(
        2
    )


    # Display model accuracy cards
    st.subheader(
        "Model Accuracy"
    )


    # Create three columns
    col1, col2, col3 = st.columns(
        3
    )


    # Display SVM accuracy
    with col1:

        st.metric(
            "SVM",
            f"{svm_accuracy * 100:.2f}%"
        )


    # Display Logistic Regression accuracy
    with col2:

        st.metric(
            "Logistic Regression",
            f"{log_accuracy * 100:.2f}%"
        )


    # Display KNN accuracy
    with col3:

        st.metric(
            "KNN",
            f"{knn_accuracy * 100:.2f}%"
        )


    # Display accuracy comparison table
    st.subheader(
        "Accuracy Comparison"
    )


    # Show model accuracy values
    st.dataframe(
        comparison_data[
            [
                "Model",
                "Accuracy (%)"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


    # Display accuracy visualization
    st.subheader(
        "Accuracy Visualization"
    )


    # Create the accuracy chart
    fig, ax = plt.subplots(
        figsize=(10, 5)
    )


    # Create a bar chart
    sns.barplot(
        data=comparison_data,
        x="Model",
        y="Accuracy (%)",
        ax=ax
    )


    # Set the y-axis range
    ax.set_ylim(
        0,
        100
    )


    # Set chart labels
    ax.set_ylabel(
        "Accuracy (%)"
    )

    ax.set_xlabel(
        "Model"
    )


    # Display the chart
    st.pyplot(
        fig
    )


    # Close the chart
    plt.close(
        fig
    )


    # Display classification reports
    st.subheader(
        "Detailed Classification Reports"
    )


    # Create tabs for the three models
    tab1, tab2, tab3 = st.tabs(
        [
            "SVM",
            "Logistic Regression",
            "KNN"
        ]
    )


    # Display SVM classification report
    with tab1:

        svm_report = classification_report(
            Y_test,
            SVM_predict,
            output_dict=True,
            zero_division=0
        )


        st.dataframe(
            pd.DataFrame(
                svm_report
            ).transpose(),
            use_container_width=True
        )


    # Display Logistic Regression classification report
    with tab2:

        log_report = classification_report(
            Y_test,
            LOG_predict,
            output_dict=True,
            zero_division=0
        )


        st.dataframe(
            pd.DataFrame(
                log_report
            ).transpose(),
            use_container_width=True
        )


    # Display KNN classification report
    with tab3:

        knn_report = classification_report(
            Y_test,
            KNN_predict,
            output_dict=True,
            zero_division=0
        )


        st.dataframe(
            pd.DataFrame(
                knn_report
            ).transpose(),
            use_container_width=True
        )


    # Display confusion matrix comparison
    st.subheader(
        "Confusion Matrix Comparison"
    )


    # Create tabs for confusion matrices
    tab1, tab2, tab3 = st.tabs(
        [
            "SVM",
            "Logistic Regression",
            "KNN"
        ]
    )


    # Get all glass type labels
    labels = sorted(
        Y.unique()
    )


    # Display SVM confusion matrix
    with tab1:

        fig, ax = plt.subplots()


        sns.heatmap(
            confusion_matrix(
                Y_test,
                SVM_predict,
                labels=labels
            ),
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )


        ax.set_xlabel(
            "Predicted"
        )

        ax.set_ylabel(
            "Actual"
        )


        st.pyplot(
            fig
        )


        plt.close(
            fig
        )


    # Display Logistic Regression confusion matrix
    with tab2:

        fig, ax = plt.subplots()


        sns.heatmap(
            confusion_matrix(
                Y_test,
                LOG_predict,
                labels=labels
            ),
            annot=True,
            fmt="d",
            cmap="Greens",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )


        ax.set_xlabel(
            "Predicted"
        )

        ax.set_ylabel(
            "Actual"
        )


        st.pyplot(
            fig
        )


        plt.close(
            fig
        )


    # Display KNN confusion matrix
    with tab3:

        fig, ax = plt.subplots()


        sns.heatmap(
            confusion_matrix(
                Y_test,
                KNN_predict,
                labels=labels
            ),
            annot=True,
            fmt="d",
            cmap="Oranges",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )


        ax.set_xlabel(
            "Predicted"
        )

        ax.set_ylabel(
            "Actual"
        )


        st.pyplot(
            fig
        )


        plt.close(
            fig
        )


    # Find the model with the highest accuracy
    best_model = comparison_data.loc[
        comparison_data["Accuracy"].idxmax(),
        "Model"
    ]


    # Get the highest accuracy
    best_accuracy = comparison_data["Accuracy"].max()


    # Display the best-performing model
    st.success(
        f"Best performing model: "
        f"{best_model} with an accuracy of "
        f"{best_accuracy * 100:.2f}%."
    )


# Display the SVM prediction page
elif page == "⚙️ SVM":

    # Display the page title
    st.title(
        "⚙️ Support Vector Machine"
    )


    # Display the page description
    st.write(
        "Predict the type of glass using the trained SVM model."
    )


    # Get user input
    input_data = get_user_input()


    # Predict when the button is clicked
    if st.button(
        "🔮 Predict Glass Type",
        type="primary"
    ):

        # Generate the prediction
        prediction = svm_model.predict(
            input_data
        )


        # Convert the prediction to an integer
        predicted_type = int(
            prediction[0]
        )


        # Display the predicted type number
        st.success(
            f"Predicted Glass Type: "
            f"{predicted_type}"
        )


        # Display the glass category
        st.info(
            f"Glass Category: "
            f"{glass_types[predicted_type]}"
        )


# Display the Logistic Regression prediction page
elif page == "📈 Logistic Regression":

    # Display the page title
    st.title(
        "📈 Logistic Regression"
    )


    # Display the page description
    st.write(
        "Predict the type of glass using the trained "
        "Logistic Regression model."
    )


    # Get user input
    input_data = get_user_input()


    # Predict when the button is clicked
    if st.button(
        "🔮 Predict Glass Type",
        type="primary"
    ):

        # Generate the prediction
        prediction = log_model.predict(
            input_data
        )


        # Convert the prediction to an integer
        predicted_type = int(
            prediction[0]
        )


        # Display the predicted type number
        st.success(
            f"Predicted Glass Type: "
            f"{predicted_type}"
        )


        # Display the glass category
        st.info(
            f"Glass Category: "
            f"{glass_types[predicted_type]}"
        )


# Display the KNN prediction page
elif page == "🔍 KNN":

    # Display the page title
    st.title(
        "🔍 K-Nearest Neighbors"
    )


    # Display the page description
    st.write(
        "Predict the type of glass using the trained KNN model."
    )


    # Get user input
    input_data = get_user_input()


    # Predict when the button is clicked
    if st.button(
        "🔮 Predict Glass Type",
        type="primary"
    ):

        # Generate the prediction
        prediction = knn_model.predict(
            input_data
        )


        # Convert the prediction to an integer
        predicted_type = int(
            prediction[0]
        )


        # Display the predicted type number
        st.success(
            f"Predicted Glass Type: "
            f"{predicted_type}"
        )


        # Display the glass category
        st.info(
            f"Glass Category: "
            f"{glass_types[predicted_type]}"
        )