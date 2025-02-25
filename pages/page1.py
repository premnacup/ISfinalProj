import streamlit as st
import pandas as pd

st.title("Dataset")
st.divider()

with open("dataset.csv", "r") as f:
    df = pd.read_csv(f)

st.header("Getting the dataset")
text = "For the first step I'll be geting a dataset which I got the dataset from [here](https://www.kaggle.com/datasets/ahmedmohamed2003/cafe-sales-dirty-data-for-cleaning-training/data)"
st.markdown(text)
st.write(df)
st.markdown("*this is the dataset at first glance*")
st.write(
    "This dataset is about Cafe Sales, as you can see from the dataset there're some errors."
)
st.write("Columns Description")
columns = df.columns
print(columns)
description = pd.DataFrame(
    {
        "Column Name": list(columns),
        "Description": [
            "A unique identifier for each transaction. Always present and unique.",
            'The name of the item purchased. May contain missing or invalid values (e.g., "ERROR").',
            "The quantity of the item purchased. May contain missing or invalid values.",
            "The price of a single unit of the item. May contain missing or invalid values.",
            "The total amount spent on the transaction. Calculated as Quantity * Price Per Unit.",
            'The method of payment used. May contain missing or invalid values (e.g., None, "UNKNOWN").',
            "The location where the transaction occurred. May contain missing or invalid values.",
            "The date of the transaction. May contain missing or incorrect values.",
        ],
        "Example Value": [
            "TXN_1234567",
            "Coffee, Sandwich",
            "1, 3, UNKNOWN",
            "2.00, 4.00",
            "8.00, 12.00",
            "Cash, Credit Card",
            "In-store, Takeaway",
            "2023-01-01",
        ],
    }
)
st.table(description)

st.header("Cleaning the dataset")
st.write(df.head())
