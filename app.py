import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("model.pkl")

# Load the dataset for feature reference
data = pd.read_csv("final_data.csv")
feature_names = list(data.columns[2:-1])  # First Column is index, Second Column is Year & Last column is the target

# Streamlit app title
st.title("Customer Churn Prediction for the Telecommunication Industry")

st.markdown("""
## Description of Columns

- Gender - For Female : Enter 0, For Male: Enter 1

For No: Enter 0, For Yes: Enter 1
            
- Senior Citizen - Is this person a Senior Citizen?
- Partner - Is this person married?
- Dependents - Is this person dependent on someone else in the family?
- Tenure - Plan is actived for how many months?
- Phone Service - Opted for Calling Service?
- Multiple Lines - Has the customer opted for more than 1 numbers?
- Internet Service - Has the customer opted for internet service?	
- Tech Support - Is tech support good?
            	
- Payment Method - For Bank Transfer: Enter 0, For Credit Card: Enter 1, For Electronic Check: Enter 2, For Mail Transfer: Enter 3	
- Monthly Charges - How much does the customer pay per month?
- Total Charges - Total charges incurred by the customer
""")
st.markdown("""
### Enter Customer Details
Provide input values for the features below to predict whether the customer is at risk of leaving.
""")

# Create input fields for each feature
inputs = {}
for feature in feature_names:
    inputs[feature] = st.number_input(f"Enter {feature}", value=0.0)

# Convert inputs to DataFrame for prediction
input_data = pd.DataFrame([inputs])

# Prediction button
if st.button("Predict"):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.error("The customer is going to leave.")
    elif prediction[0] == 0:
        st.success("The customer will stay.")
        st.balloons()
