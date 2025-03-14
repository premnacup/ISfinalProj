import streamlit as st
import pandas as pd
import pickle
import random

@st.cache_resource
def load_model(model_path):
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocessInput(df):
    mapping = {
        "Gender": {"Male": 1, "Female": 0},
        "Married": {"Yes": 1, "No": 0},
        "Dependents": {"0": 1, "1": 2, "2": 3, "3+": 4},
        "Education": {"Graduate": 1, "Not Graduate": 0},
        "Self_Employed": {"Yes": 1, "No": 0},
        "Coapplicant": {"Yes": 1, "No": 0},
        "loan_History": {"Yes": 1, "No": 0},
        "Area": {"Urban": 1, "Semiurban": 2, "Rural": 3},
    }

    for col, mapping_dict in mapping.items():
        df[col] = df[col].map(mapping_dict)

    return df.astype(int).values

def generate_random_inputs_ML():
    return {
        "Gender": random.choice(["Male", "Female"]),
        "Married": random.choice(["Yes", "No"]),
        "Dependents": random.choice(["0", "1", "2", "3+"]),
        "Education": random.choice(["Graduate", "Not Graduate"]),
        "Self_Employed": random.choice(["Yes", "No"]),
        "Income(dollar)": random.randint(1000, 10000),
        "Coapplicant": random.choice(["Yes", "No"]),
        "Loan_Amount": random.randint(1000, 50000),
        "Term(month)": random.choice([12, 36, 60, 120, 180, 240, 360]),
        "loan_History": random.choice(["Yes", "No"]),
        "Area": random.choice(["Rural", "Semiurban", "Urban"]),
    }

st.title("Machine Learning Demo")

svmTab, dtTab = st.tabs(["Support Vector Machine", "Decision Tree"])

with svmTab:
    st.header("Support Vector Machine Model")

    model = load_model("./model/SVM/SVM_model.pkl")

    if model is None:
        st.error("Model could not be loaded. Please check the file path.")
    else:
        if st.session_state.get("random_inputs_ML", None) is None:
            random_inputs_ML = generate_random_inputs_ML()
            if st.button("Generate Random Inputs"):
                st.session_state.random_inputs_ML = random_inputs_ML
        else:
            random_inputs_ML = st.session_state.get("random_inputs_ML", 1)

        st.subheader("Personal Information")
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"],
            index=0 if not random_inputs_ML else ["Male", "Female"].index(random_inputs_ML["Gender"]),
            key="gender"
        )

        married = st.selectbox(
            "Married",
            ["Yes", "No"],
            index=0 if not random_inputs_ML else ["Yes", "No"].index(random_inputs_ML["Married"]),
            key="married"
        )

        dependents = st.selectbox(
            "Dependents",
            ["0", "1", "2", "3+"],
            index=0 if not random_inputs_ML else ["0", "1", "2", "3+"].index(random_inputs_ML["Dependents"]),
            key="dependents"
        )

        graduate = st.selectbox(
            "Education",
            ["Graduate", "Not Graduate"],
            index=0 if not random_inputs_ML else ["Graduate", "Not Graduate"].index(random_inputs_ML["Education"]),
            key="graduate"
        )

        self_employed = st.selectbox(
            "Self-Employed",
            ["Yes", "No"],
            index=0 if not random_inputs_ML else ["Yes", "No"].index(random_inputs_ML["Self_Employed"]),
            key="self_employed"
        )

        st.subheader("Financial Information")
        income = st.number_input(
            "Income (dollars)",
            min_value=0,
            value=0 if not random_inputs_ML else random_inputs_ML["Income(dollar)"],
            key="income"
        )

        coapplicant = st.selectbox(
            "Co-applicant",
            ["Yes", "No"],
            index=0 if not random_inputs_ML else ["Yes", "No"].index(random_inputs_ML["Coapplicant"]),
            key="coapplicant"
        )

        loan_amount = st.number_input(
            "Loan Amount (dollars)",
            min_value=0,
            value=0 if not random_inputs_ML else random_inputs_ML["Loan_Amount"],
            key="loan_amount"
        )

        loan_term = st.number_input(
            "Loan Term (months)",
            min_value=0,
            max_value=480,
            value=0 if not random_inputs_ML else random_inputs_ML["Term(month)"],
            key="loan_term"
        )

        credit_history = st.selectbox(
            "Credit History",
            ["Yes", "No"],
            index=0 if not random_inputs_ML else ["Yes", "No"].index(random_inputs_ML["loan_History"]),
            key="credit_history"
        )

        property_area = st.selectbox(
            "Property Area",
            ["Rural", "Semiurban", "Urban"],
            index=0 if not random_inputs_ML else ["Rural", "Semiurban", "Urban"].index(random_inputs_ML["Area"]),
            key="property_area"
        )

        input_data = pd.DataFrame(
            {
                "Gender": [gender],
                "Married": [married],
                "Dependents": [dependents],
                "Education": [graduate],
                "Self_Employed": [self_employed],
                "Income(dollar)": [income],
                "Coapplicant": [coapplicant],
                "Loan_Amount": [loan_amount],
                "Term(month)": [loan_term],
                "loan_History": [credit_history],
                "Area": [property_area],
            }
        )

        if st.button("Predict", use_container_width=True):
            st.write("### Your Input Data")
            st.dataframe(input_data, hide_index=True)
            prediction = model.predict(preprocessInput(input_data))
            if prediction[0] == 1:
                st.write("### Prediction: Approved")
            else:
                st.write("### Prediction: Not Approved")


        if st.button("Random Predict 5 times", use_container_width=True):
            for _ in range(5):
                random_inputs_ML = generate_random_inputs_ML()
                X_test = pd.DataFrame([random_inputs_ML])
                st.dataframe(X_test, hide_index=True)
                prediction = model.predict(preprocessInput(X_test))
                if prediction[0] == 1:
                    st.write("### Prediction: Approved")
                else:
                    st.write("### Prediction: Not Approved")

        if st.button("Reset Input", use_container_width=True):
            st.session_state.clear()
            st.rerun()


with dtTab:
    st.header("Decision Tree Model")

    model = load_model("./model/DT/DecisionTree_model.pkl")

    if model is None:
        st.error("Model could not be loaded. Please check the file path.")
    else:
        if st.button("Generate Random Inputs", key="generateRandomInputsDT"):
            random_inputs_ML = generate_random_inputs_ML()
            st.session_state.random_inputs_ML = random_inputs_ML
        else:
            random_inputs_ML = st.session_state.get("random_inputs_ML", None)

        st.subheader("Personal Information")
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"],
            index=0 if not random_inputs_ML else ["Male", "Female"].index(random_inputs_ML["Gender"]),
            key="genderDT"
        )

        married = st.selectbox(
            "Married",
            ["Yes", "No"],
            index=0 if not random_inputs_ML else ["Yes", "No"].index(random_inputs_ML["Married"]),
            key="marriedDT"
        )

        dependents = st.selectbox(
            "Dependents",
            ["0", "1", "2", "3+"],
            index=0 if not random_inputs_ML else ["0", "1", "2", "3+"].index(random_inputs_ML["Dependents"]),
            key="dependentsDT"
        )

        graduate = st.selectbox(
            "Education",
            ["Graduate", "Not Graduate"],
            index=0 if not random_inputs_ML else ["Graduate", "Not Graduate"].index(random_inputs_ML["Education"]),
            key="graduateDT"
        )

        self_employed = st.selectbox(
            "Self-Employed",
            ["Yes", "No"],
            index=0 if not random_inputs_ML else ["Yes", "No"].index(random_inputs_ML["Self_Employed"]),
            key="self_employedDT"
        )

        st.subheader("Financial Information")
        income = st.number_input(
            "Income (dollars)",
            min_value=0,
            value=0 if not random_inputs_ML else random_inputs_ML["Income(dollar)"],
            key="incomeDT"
        )

        coapplicant = st.selectbox(
            "Co-applicant",
            ["Yes", "No"],
            index=0 if not random_inputs_ML else ["Yes", "No"].index(random_inputs_ML["Coapplicant"]),
            key="coapplicantDT"
        )

        loan_amount = st.number_input(
            "Loan Amount (dollars)",
            min_value=0,
            value=0 if not random_inputs_ML else random_inputs_ML["Loan_Amount"],
            key="loan_amountDT"
        )

        loan_term = st.number_input(
            "Loan Term (months)",
            min_value=0,
            max_value=480,
            value=0 if not random_inputs_ML else random_inputs_ML["Term(month)"],
            key="loan_termDT"
        )

        credit_history = st.selectbox(
            "Credit History",
            ["Yes", "No"],
            index=0 if not random_inputs_ML else ["Yes", "No"].index(random_inputs_ML["loan_History"]),
            key="credit_historyDT"
        )

        property_area = st.selectbox(
            "Property Area",
            ["Rural", "Semiurban", "Urban"],
            index=0 if not random_inputs_ML else ["Rural", "Semiurban", "Urban"].index(random_inputs_ML["Area"]),
            key="property_areaDT"
        )

        input_data = pd.DataFrame(
            {
                "Gender": [gender],
                "Married": [married],
                "Dependents": [dependents],
                "Education": [graduate],
                "Self_Employed": [self_employed],
                "Income(dollar)": [income],
                "Coapplicant": [coapplicant],
                "Loan_Amount": [loan_amount],
                "Term(month)": [loan_term],
                "loan_History": [credit_history],
                "Area": [property_area],
            }
        )

        if st.button("Predict", use_container_width=True, key="predictDT"):
            st.write("### Your Input Data")
            st.dataframe(input_data, hide_index=True)
            prediction = model.predict(preprocessInput(input_data))
            if prediction[0] == 1:
                st.write("### Prediction: Approved")
            else:
                st.write("### Prediction: Not Approved")


        if st.button("Random Predict 5 times", use_container_width=True,key="randomPredictDT"):
            for _ in range(5):
                random_inputs_ML = generate_random_inputs_ML()
                X_test = pd.DataFrame([random_inputs_ML])
                st.dataframe(X_test, hide_index=True)
                prediction = model.predict(preprocessInput(X_test))
                if prediction[0] == 1:
                    st.write("### Prediction: Approved")
                else:
                    st.write("### Prediction: Not Approved")

        if st.button("Reset Input", use_container_width=True,key="resetInputDT"):
            st.session_state.clear()
            st.rerun()
