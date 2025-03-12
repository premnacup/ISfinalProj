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

df = pd.read_csv("./data/dataset.csv")
df2 = pd.read_csv("./data/dataset2.csv")

def preprocess_dataset(data):
    data["Status"] = data["Status"].map({"Y": 1, "N": 0})
    data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})
    data["Married"] = data["Married"].map({"Yes": 1, "No": 0})
    data["Education"] = data["Education"].map({"Graduate": 1, "Not Graduate": 0})
    data["Self_Employed"] = data["Self_Employed"].map({"Yes": 1, "No": 0})
    data["Area"] = data["Area"].map({"Urban": 1, "Semiurban": 2, "Rural": 3})
    data["Coapplicant"] = data["Coapplicant"].map({"Yes": 1, "No": 0})
    data["Dependents"] = data["Dependents"].map({"0": 1, "1": 2, "2": 3, "3+": 4})

    data = data.dropna()

    return data.astype(int)

def preprocess_dataset2(data):
    data.dropna(inplace = True)
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    trafficStatus = ['low', 'normal', 'heavy', 'high']

    data['Time'] = pd.to_datetime(data['Time'], format="%I:%M:%S %p")
    data['Hour'] = data['Time'].dt.hour
    data['Minute'] = data['Time'].dt.minute
    data['Second'] = data['Time'].dt.second
    data.drop(columns=['Time'], inplace=True)

    data['Day of the week'] = data['Day of the week'].map(lambda x: days.index(x)).astype(int)
    data['Traffic Situation'] = data['Traffic Situation'].map(lambda x: trafficStatus.index(x)).astype(int)

    return data.astype(int)

df.dropna(inplace=True)
dfSVM = preprocess_dataset(df.copy())
dfDT = preprocess_dataset(df.copy())
df2 = preprocess_dataset2(df2.copy())

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
    st.write(dfSVM)

    st.write(
        """
        Now we can start to develop the model first we need to split the features and the target, then we need to scale the features since the values are very big
        """
    )

    X = dfSVM.drop("Status", axis=1)
    y = dfSVM["Status"]

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
        """
    )

    st.dataframe(x_scaled, hide_index=True)

    st.write(
        """
        The next step will be splitting the dataset into training and testing set, then we can start to develop the model and save the model
        """
    )

    X_train, X_test, y_train, y_test = train_test_split(
        x_scaled, y, test_size=0.3, random_state=42
    )

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
        pickle.dump(svm_classifier, open('SVM_model.pkl', 'wb'))
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
    st.divider()
    st.header("Getting to know the algorithm first")
    st.write(
        """
        <p style="text-align: justify;">
        The second model I used for this dataset is the Decision Tree.
        A decision tree is a graphical representation of different options for solving a problem and shows how various factors are related. 
        It has a hierarchical tree structure that starts with one main question at the top, called a node, which further branches out into different possible outcomes where:

        - **Root Node**: The starting point that represents the entire dataset.
        - **Branches**: Lines that connect nodes, showing the flow from one decision to another.
        - **Internal Nodes**: Points where decisions are made based on the input features.
        - **Leaf Nodes**: Terminal nodes at the end of branches that represent final outcomes or predictions.

        </p>
        """,
        unsafe_allow_html=True,
    )
    st.image("https://media.geeksforgeeks.org/wp-content/uploads/20250107141217134593/Decision-Tree.webp")
    st.write(
        """
        The algorithm works by creating a tree structure that represents the decision process.
        Which the decision progress is kind of similar to checking a condition in if else statement.
        Starting to the root node to the leaf node. The algorithm will continue to check the condition until 
        it reaches the leaf node. Once you reach the leaf node, the algorithm will make a prediction based on the value of the node.
        """
    )

    st.header("Developing the model")
    st.write(
        "After finished preparing the dataset we can now develop the model from the dataset now you can see that all features are now numerical value"
    )
    st.write(dfDT)

    st.write(
        """
        Now we can start to develop the model first we need to split the features and the target, then we need to scale the features since the values are very big
        """
    )

    X_dt = dfDT.drop("Status", axis=1)
    y_dt = dfDT["Status"]
    
    x_scaled_dt = pd.DataFrame(scale.fit_transform(X_dt), columns=X_dt.columns)

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
    st.dataframe(x_scaled_dt, hide_index=True)

    st.write(
        """
        The next step will be splitting the dataset into training and testing set, then we can start to develop the model and save the model
        """
    )

    X_train, X_test, y_train, y_test = train_test_split(
        x_scaled_dt, y_dt, test_size=0.3, random_state=42
    )

    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X_train, y_train)
    dt_predictions = dt_classifier.predict(X_test)

    st.code(
        """
        # split the dataset
        X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3, random_state=42)

        # develop the model
        dt_classifier = DecisionTreeClassifier(random_state=42)
        dt_classifier.fit(X_train, y_train)
        dt_predictions = dt_classifier.predict(X_test)

        # save the model
        pickle.dump(dt_classifier, open('DT/DecisionTree_model.pkl', 'wb'))
        """
    )

    st.write(
        """
        Next is evaluating the model
        """
    )

    accuracy = accuracy_score(y_test, dt_predictions)
    precision = precision_score(y_test, dt_predictions, average="weighted")
    recall = recall_score(y_test, dt_predictions, average="weighted")
    f1 = f1_score(y_test, dt_predictions, average="weighted")
    cm = confusion_matrix(y_test, dt_predictions)

    st.code(
        """
        accuracy = accuracy_score(y_test, dt_predictions)
        precision = precision_score(y_test, dt_predictions, average="weighted")
        recall = recall_score(y_test, dt_predictions, average="weighted")
        f1 = f1_score(y_test, dt_predictions, average="weighted")
        cm = confusion_matrix(y_test, dt_predictions)   
        """
    )

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
    st.image("./model/DT/confusionMatrix.png", use_container_width=True)
with tabNN:
    st.title("Feedforward Neural Network (FNN)")
    st.divider()

    st.header("Getting to know the algorithm first")
    st.write(
        """
        For my Neural Network model I chose Feedforward Neural Network (FNN). 
        A Feedforward Neural Network (FNN) is neural network 
        where connections between the nodes don't form cycles. The network consists 
        of an input layer, one or more hidden layers, and an output layer. Information 
        flows in one direction—from input to output.
        """
    )

    st.write(
        """
        Structure of a Feedforward Neural Network

        - **Input Layer**: The input layer consists of neurons that receive the input data. Each neuron in the input layer represents a feature of the input data.
        - **Hidden Layers**: One or more hidden layers are placed between the input and output layers. These layers are responsible for learning the complex patterns in the data. Each neuron in a hidden layer applies a weighted sum of inputs followed by a non-linear activation function.
        - **Output Layer**: The output layer provides the final output of the network. The number of neurons in this layer corresponds to the number of classes in a classification problem or the number of outputs in a regression problem.

        Each connection between neurons in these layers has an associated weight that is adjusted during the training process to minimize the error in predictions.
        """
    )
    st.image("https://media.geeksforgeeks.org/wp-content/uploads/20240601001059/FNN-768.jpg")

    st.header("Developing the model")
    st.write("After finished preparing the dataset we can now develop the model from the dataset now you can see that all features are now numerical value")
    st.write(df2)

    st.write("First is to import all package")
    st.code(
        """
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler
        import matplotlib.pyplot as plt
        import pickle
        from tensorflow.keras import Input
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.utils import to_categorical
        """
    )

    st.write("Now we can start to develop the model first we need to split the features and the target, then we need to scale the features")
    st.code(
        """
        # split the features and the target
        X = df.drop(columns=['Traffic Situation'])
        y = df['Traffic Situation']

        # scale the features using MinMaxScaler
        scaler = MinMaxScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        y_one_hot = to_categorical(y, num_classes=len(trafficStatus))
        """
    )

    st.write("The next step will be splitting the dataset into training and testing set, then we can start to develop the model and save the model")
    st.code(
        """
        # convert the target to one-hot encoding
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
        y_train = to_categorical(y_train, num_classes=4)
        y_test = to_categorical(y_test, num_classes=4)

        # develop the model
        model = Sequential([
            Input(shape=(X_train.shape[1],)),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(4, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
        test_loss, test_acc = model.evaluate(X_test, y_test)

        # save the model
        pickle.dump(model, open('NN/NN_model.pkl', 'wb'))
        """
    )
    st.write("The accuracy of the model is shown below (89%-90%)")
    st.image("./model/NN/accuracy.png", use_container_width=True)
