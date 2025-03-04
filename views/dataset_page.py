import streamlit as st
import pandas as pd

tabML, tabNN = st.tabs(["Machine Learning", "Neural Network"])
with tabML:
    st.title("Machine Learning Dataset")
    st.divider()

    with open("./data/dataset.csv", "r") as f:
        df = pd.read_csv(f)

    st.header("Getting the dataset")
    text = "For the first step I'll be geting a dataset which I got the dataset from older assignment [(download here)](https://github.com/premnacup/ISfinalProj/blob/main/data/dataset.csv)."
    st.markdown(text)
    st.write(df)
    st.markdown("*this is the dataset at first glance*")
    st.write(
        "This dataset is about Loan Approval, as you can see from the dataset there're some errors."
    )
    st.markdown(
        """
        <h2> Columns Description </h2>
        """,
        unsafe_allow_html=True,
    )
    columns = df.columns
    description = pd.DataFrame(
        {
            "Column Name": list(columns),
            "Description": [
                "Specifies the gender of the loan applicant.",
                "Indicates whether the applicant is married.",
                "Represents the number of dependents financially reliant on the applicant.",
                "Describes the applicant's highest level of education.",
                "Specifies whether the applicant is self-employed.",
                "The annual income of the primary applicant, measured in dollars.",
                "The annual income of a co-applicant, if applicable.",
                "The total loan amount requested by the applicant, measured in dollars.",
                "The loan repayment period specified in months.",
                "Indicates whether the applicant has a recorded credit history.",
                "The type of residential area where the applicant resides.",
                "The final decision on the loan application.",
            ],
            "Example": [
                df[i].unique() if df[i].dtypes == "object" else df[i].iloc[0]
                for i in df.columns
            ],
        }
    )
    st.dataframe(description, hide_index=True)

    st.header("Cleaning the dataset")
    st.code(
        """
        # reading the dataset
        df = pd.read_csv('../data/dataset.csv')
        
        # checking the dataset columns and check for null value
        df.columns
        df.isnull().any()

        # drop the null value
        df.dropna(inplace = True)
        """,
        language="python",
    )

    st.write("After dropping the null value")
    df.dropna(inplace=True)

    st.write(df)

    st.code(
        """
        # describe the dataset to check for outlier
        df.describe(include = 'all')
        
        # encode the dataset into numerical value
        df['Status'] = df['Status'].map({'Y': 1, 'N': 0})
        df['Gender'] = df['Gender'].map({'Male' : 1, 'Female' : 0})
        df['Married'] = df['Married'].map({'Yes' : 1, 'No' : 0})
        df['Education'] = df['Education'].map({'Graduate' : 1, 'Not Graduate' : 0})
        df['Self_Employed'] = df['Self_Employed'].map({'Yes' : 1, 'No' : 0})
        df['Area'] = df['Area'].map({'Urban' : 1, 'Semiurban' : 2, 'Rural' : 3})
        df['Coapplicant'] = df['Coapplicant'].map({'Yes' : 1, 'No' : 0})
        df['Dependents'] = df['Dependents'].map({'0' : 1,'1' : 2,'2':3,'3+' : 4})

        # drop the null value again to avoid error and convert the dataset into integer
        df = df.dropna()
        df.astype(int)
        """,
        language="python",
    )

    st.write("This is df.describe")
    st.write(df.describe(include="all"))

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

    st.write("After encoding the dataset")
    st.write(df)

    st.write(
        "Now we can see the dataset is clean and ready to be used for machine learning"
    )

with tabNN:
    st.title("Neural Network")
    st.divider()
