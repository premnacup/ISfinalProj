import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.preprocessing import MinMaxScaler

with open("./data/dataset.csv", "r") as f:
    df = pd.read_csv(f)

dfSVM = df
dfDT = df
df.dropna(inplace=True)
df["Status"] = df["Status"].map({"Y": 1, "N": 0})
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
df["Married"] = df["Married"].map({"Yes": 1, "No": 0})
df["Education"] = df["Education"].map({"Graduate": 1, "Not Graduate": 0})
df["Self_Employed"] = df["Self_Employed"].map({"Yes": 1, "No": 0})
df["Area"] = df["Area"].map({"Urban": 1, "Semiurban": 2, "Rural": 3})
df["Coapplicant"] = df["Coapplicant"].map({"Yes": 1, "No": 0})
df["Dependents"] = df["Dependents"].map({"0": 1, "1": 2, "2": 3, "3+": 4})

df = df.dropna()
df.astype(int)


tabSVM, tabDT, tabNN = st.tabs(
    ["Machine Learning (SVM)", "Machine Learning (Decision Tree)", "Neural Network"]
)
with tabSVM:
    st.title("Machine Learning (SVM)")
    st.divider()
    st.header("Getting to know the algorithm first")
    st.write(
        """
        <p style="text-align: justify;">
        The first model I used for this dataset is SVM (Support Vector Machine).
        SVM is a supervised learning algorithm that uses a hyperplane to separate the data into different classes.
        The goal of SVM is to find the best hyperplane that maximizes the margin between the classes.
        The margin is the distance between the hyperplane and the closest points in each class.
        The margin is important because it is the distance between the hyperplane and the closest points in each class.
        The margin is also important because it is the distance between the hyperplane and the closest points in each class.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.image(
        "https://media.geeksforgeeks.org/wp-content/uploads/20231109124312/Hinge-loss-(2).png"
    )

    st.header("Developing the model")
    st.write(
        "After finished preparing the dataset we can now develop the model from the dataset now you can see that all features are now numerical value"
    )
    st.write(df)

    st.write(
        """
        Now we can start to develop the model first we need to split the features and the target, then we need to scale the features since the values are very big
        """
    )

    X = df.drop("Status", axis=1)
    y = df["Status"]

    scale = MinMaxScaler()
    x_scaled = pd.DataFrame(scale.fit_transform(X), columns=X.columns)

    st.code(
        """
        # split the features and the target
        X = df.drop('Status', axis=1)
        y = df['Status']

        # scale the features using MinMaxScaler
        scale = MinMaxScaler()
        x_scaled = pd.DataFrame(scale.fit_transform(X), columns=X.columns)
        x_scaled
        """
    )

    st.write(
        """
        The next step will be splitting the dataset into training and testing set, then we can start to develop the model and save the model
        """
    )

    # split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        x_scaled, y, test_size=0.3, random_state=42
    )

    # develop the model
    svm_classifier = SVC(kernel="linear")
    svm_classifier.fit(X_train, y_train)
    svm_predictions = svm_classifier.predict(X_test)

    st.code(
        """
        # split the dataset
        X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3, random_state=42)

        # develop the model
        svm_classifier = SVC(kernel='linear')
        svm_classifier.fit(X_train, y_train)
        svm_predictions = svm_classifier.predict(X_test)

        # save the model
        pickle.dump(svm_predictions, open('SVM_model.pkl', 'wb'))
        """
    )

    st.write(
        """
        Next is evaluating the model
        """
    )

    accuracy = accuracy_score(y_test, svm_predictions)
    precision = precision_score(y_test, svm_predictions, average="weighted")
    recall = recall_score(y_test, svm_predictions, average="weighted")
    f1 = f1_score(y_test, svm_predictions, average="weighted")
    cm = confusion_matrix(y_test, svm_predictions)

    st.code(
        """
        accuracy = accuracy_score(y_test, svm_predictions)
        precision = precision_score(y_test, svm_predictions, average="weighted")
        recall = recall_score(y_test, svm_predictions, average="weighted")
        f1 = f1_score(y_test, svm_predictions, average="weighted")
        cm = confusion_matrix(y_test, svm_predictions)      
        """
    )

    # Creating a DataFrame
    df = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1 score"],
            "Value": [accuracy, precision, recall, f1],
        }
    )

    # Custom HTML + CSS for styling
    html_code = f"""
    <style>
        .result-container {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            width: 100%;
            margin: auto;
            text-align: center;
            color: #2c3e50;
        }}
        .result-container h3 {{
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        .result-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .result-table th, .result-table td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        .result-table th {{
            background: #3498db;
            color: white;
        }}
    </style>

    <div class="result-container">
        <h3>Model Evaluation Results</h3>
        <table class="result-table">
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr><td>Accuracy</td><td>{accuracy:.2%}</td></tr>
            <tr><td>Precision</td><td>{precision:.2%}</td></tr>
            <tr><td>Recall</td><td>{recall:.2%}</td></tr>
            <tr><td>F1 Score</td><td>{f1:.2%}</td></tr>
        </table>
    </div>
    """

    st.markdown(html_code, unsafe_allow_html=True)

    st.write("### Confusion Matrix:")
    st.image("./model/SVM/confusionMatrix.png", use_container_width=True)

with tabDT:
    st.title("Decision Tree")
with tabNN:
    st.title("Neural Network")
    st.divider()
