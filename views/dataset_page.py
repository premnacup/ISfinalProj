import streamlit as st
import pandas as pd

tabML, tabNN = st.tabs(["Machine Learning", "Neural Network"])
with tabML:
    st.title("Machine Learning Dataset")
    st.divider()

    with open("./data/dataset.csv", "r") as f:
        df = pd.read_csv(f)

    st.header("Getting the dataset")
    st.write(
        """
        For the first step I'll be geting a dataset which I got the dataset from older assignment.
        \n[download here](https://github.com/premnacup/ISfinalProj/blob/main/data/dataset.csv)
        """
    )
    st.write(df)
    st.markdown("*this is the dataset at first glance*")
    st.write(
        "This dataset is about Loan Approval, as you can see from the dataset there're some errors."
    )

    st.header("Columns Description")
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

    with open("./data/dataset2.csv", "r") as f:
        df2 = pd.read_csv(f)

    st.header("Getting the dataset")
    st.write(
        """
        For the first step I'll be geting a dataset which I got the dataset from kaggle.com and used AI to make the data dirty to allow data preparation and data cleaning.
        \n[download here](https://www.kaggle.com/datasets/hasibullahaman/traffic-prediction-dataset?select=Traffic.csv)
        """
    )
    st.write(df2)
    st.markdown("*this is the dataset at first glance*")
    st.write(
        "This dataset is about traffic congestion, as you can see from the dataset there're some errors."
    )

    st.header("Columns Description")
    columns_ = df2.columns
    description_ = pd.DataFrame(
        {
            "Column Name": list(columns_),
            "Description": [
                "Time of the day when the data was recorded.",
                "Day of the month corresponding to the recorded data.",
                "Day of the week when the data was collected.",
                "Number of cars counted on that particular day.",
                "Number of bikes recorded on that day.",
                "Number of buses observed on that day.",
                "Number of trucks recorded on that day.",
                "Total count of all types of vehicles combined.",
                "Traffic situation classification for the day."
            ],
            "Example": [
                str(df2[i].unique().tolist()) if df2[i].dtypes == "object" else str(df2[i].iloc[0])
                for i in df2.columns
            ]
        }
    )
    st.dataframe(description_, hide_index=True)

    st.header("Cleaning the dataset")
    st.code(
        """
        # reading the dataset
        df = pd.read_csv('../data/dataset2.csv')
        
        # checking the dataset columns and check for null value
        df.columns
        df.isnull().any()

        # drop the null value
        df.dropna(inplace = True)
        """,
        language="python",
    )

    st.write("After dropping the null value")
    df2.dropna(inplace=True)
    st.write(df2)

    st.write("Encode the data and doing feature engineering")
    st.code(
        """
        # prepare days and status for encoding
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        trafficStatus = ['low', 'normal', 'heavy', 'high']

        # extract hour, minute, and second from df['Time] and remove the time column
        df['Time'] = pd.to_datetime(df['Time'], format="%I:%M:%S %p")
        df['Hour'] = df['Time'].dt.hour
        df['Minute'] = df['Time'].dt.minute
        df['Second'] = df['Time'].dt.second
        df.drop(columns=['Time'], inplace=True)

        # encode into the prepared value
        df['Day of the week'] = df['Day of the week'].map(lambda x: days.index(x)).astype(int)
        df['Traffic Situation'] = df['Traffic Situation'].map(lambda x: trafficStatus.index(x)).astype(int)
        """
    )
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    trafficStatus = ['low', 'normal', 'heavy', 'high']

    df2['Time'] = pd.to_datetime(df2['Time'], format="%I:%M:%S %p")
    df2['Hour'] = df2['Time'].dt.hour
    df2['Minute'] = df2['Time'].dt.minute
    df2['Second'] = df2['Time'].dt.second
    df2.drop(columns=['Time'], inplace=True)

    df2['Day of the week'] = df2['Day of the week'].map(lambda x: days.index(x)).astype(int)
    df2['Traffic Situation'] = df2['Traffic Situation'].map(lambda x: trafficStatus.index(x)).astype(int)

    st.write("After encoding the dataset")
    st.write(df2)

    st.write(
        "Now we can see the dataset is clean and ready to be used for training the model"
    )




