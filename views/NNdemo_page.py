import streamlit as st
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tensorflow.keras.models import load_model
from datetime import datetime
import random
import pickle

with open("./model/NN/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

def preprocessInput(data):
    data.dropna(inplace=True)
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    data['Day of the week'] = data['Day of the week'].map(lambda x: days.index(x)).astype(int)
    data['Time'] = pd.to_datetime(data['Time'], format="%H:%M:%S")
    data['Hour'] = data['Time'].dt.hour
    data['Minute'] = data['Time'].dt.minute
    data['Second'] = data['Time'].dt.second
    data.drop(columns=['Time'], inplace=True)
    return pd.DataFrame(scaler.transform(data), columns=data.columns)

def generate_random_input():
    time = datetime.today().strftime("%H:%M:%S")
    date = random.randint(1, 31)
    day = random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    car = random.randint(0, 100)
    bike = random.randint(0, 100)
    bus = random.randint(0, 50)
    truck = random.randint(0, 20)
    totalVehicle = car + bike + bus + truck
    return time, date, day, car, bike, bus, truck, totalVehicle

@st.cache_resource
def load_model_from_file():
    return load_model("./model/NN/NN_model.keras")

model = load_model_from_file()

st.title("Neural Network Demo")

if model is None:
    st.error("Model could not be loaded. Please check the file path.")
else:
    if st.button("Generate Random Input"):
        random_inputs = generate_random_input()
        st.session_state.random_inputs = random_inputs
        st.rerun()

    random_inputs = st.session_state.get("random_inputs", None)

    time = st.time_input(
        "Input time",
        value=datetime.strptime(random_inputs[0], "%H:%M:%S") if random_inputs else datetime.now()
    )
    date = st.number_input(
        "Input date (1-31)",
        min_value=1,
        max_value=31,
        value=random_inputs[1] if random_inputs else 1
    )
    day = st.selectbox(
        "Day of the week",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], 
        index=0 if not random_inputs else ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(random_inputs[2])
    )
    car = st.number_input(
        "CarCount",
        min_value=0,
        value=random_inputs[3] if random_inputs else 0
    )
    bike = st.number_input(
        "BikeCount",
        min_value=0, 
        value=random_inputs[4] if random_inputs else 0
    )
    bus = st.number_input(
        "BusCount",
        min_value=0,
        value=random_inputs[5] if random_inputs else 0
    )
    truck = st.number_input(
        "TruckCount",
        min_value=0,
        value=random_inputs[6] if random_inputs else 0
    )
    totalVehicle = car + bike + bus + truck

    input_data = pd.DataFrame({
        "Time": [time],
        "Date": [date],
        "Day of the week": [day],
        "CarCount": [car],
        "BikeCount": [bike],
        "BusCount": [bus],
        "TruckCount": [truck],
        "Total": [totalVehicle],
    })

    processed_input = preprocessInput(input_data)
    st.dataframe(processed_input, hide_index=True)

    if st.button("Predict", use_container_width=True):
        st.write("### Your Input Data")
        st.dataframe(input_data, hide_index=True)
        
        prediction = model.predict(processed_input)
        st.write(prediction)
        
        # Get the index of the highest value from the prediction
        highest_index = prediction[0].argmax()
        
        # Map the index to traffic situation levels
        traffic_levels = ["Low", "Normal", "High", "Heavy"]
        predicted_traffic = traffic_levels[highest_index]
        
        # Displaying prediction result with explanation
        st.write(f"### Prediction Result")
        st.write(f"The predicted traffic situation is: {predicted_traffic} (Score: {prediction[0][highest_index]:.2f})")

    if st.button("Reset Input", use_container_width=True):
        if "random_inputs" in st.session_state:
            del st.session_state["random_inputs"]
        st.rerun()

