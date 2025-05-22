import streamlit as st
import pandas as pd
import pickle
import xgboost
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
with open("UPI_Fraud_model.pkl", "rb") as file:
    model = pickle.load(file)

# Function to predict fraud
def predict_fraud(transaction_data):
    # Convert Date to timestamp (seconds since epoch)
    transaction_data = transaction_data.copy()
    transaction_data['Date'] = pd.to_datetime(transaction_data['Date']).astype('int64') / 10**9

    # One-hot encode categorical variables
    transaction_data = pd.get_dummies(transaction_data, columns=["Transaction_Type", "Payment_Gateway", "Transaction_State", "Merchant_Category"], drop_first=True)

    # Ensure the DataFrame has the same columns as the model was trained on
    transaction_data = transaction_data.reindex(columns=model.get_booster().feature_names, fill_value=0)

    prediction = model.predict(transaction_data)
    return prediction

# Function to visualize results
def visualize_results(df):
    # Pie chart for fraud detection results
    pie_chart = df['fraud'].value_counts().reset_index()
    pie_chart.columns = ['Fraud Status', 'Count']
    fig_pie = px.pie(pie_chart, values='Count', names='Fraud Status', title='Fraud Detection Results', color='Fraud Status', 
                     color_discrete_sequence=['#636EFA', '#EF553B'])
    st.plotly_chart(fig_pie)

    # Line chart for transaction amounts
    line_chart = df.groupby('Date')['amount'].sum().reset_index()
    fig_line = px.line(line_chart, x='Date', y='amount', title='Total Transaction Amount Over Time')
    st.plotly_chart(fig_line)

# Streamlit app layout
st.set_page_config(page_title="Safepay-AI", layout="wide")
st.title("Welcome to Safepay-AI: Revolutionizing UPI Transactions with Fraud Detection Powered by AI")
st.markdown("""
    Inspect a single transaction by adjusting the parameters on the left sidebar, 
    or upload a CSV file to check multiple transactions at once. 
    Our advanced machine learning model will process the data, 
    detect fraud, and provide detailed analysis with interactive graphs and insights for you.
""")

# Sidebar for user inputs
st.sidebar.header("Individual Transaction")
transaction_date = st.sidebar.date_input("Select Transaction Date")
transaction_type = st.sidebar.selectbox("Select Transaction Type", ["Refund", "Bank Transfer", "Subscription", "Purchase", "Investment", "Other"])
payment_gateway = st.sidebar.selectbox("Select Payment Gateway", ["SamplePay", "UPI Pay", "Dummy Bank", "Alpha Bank", "Other"])
transaction_state = st.sidebar.selectbox("Select Transaction State", ["Chhattisgarh", "Himachal Pradesh", "Mizoram", "Other"])
merchant_category = st.sidebar.selectbox("Select Merchant Category", ["Brand Vouchers and OTT", "Home delivery", "Utilities", "Other"])
transaction_amount = st.sidebar.number_input("Enter Transaction Amount upto 5,00,000(UPI Transaction Limit)", min_value=0.0, max_value=500000.0)

# File upload option
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    with st.spinner("Please wait, your data is being processed by our advanced machine learning model..."):
        df = pd.read_csv(uploaded_file)
        st.success("Data processing complete!")
        st.write("Uploaded Data:")
        st.dataframe(df)

        # Process the uploaded data to match the required format
        processed_data = pd.DataFrame({
            "Date": pd.to_datetime(df['Date']),
            "Transaction_Type": df['Transaction_Type'],
            "Payment_Gateway": df['Payment_Gateway'],
            "Transaction_State": df['Transaction_State'],
            "Merchant_Category": df['Merchant_Category'],
            "amount": df['amount']
        })

        # Make predictions on the processed data
        processed_data['fraud'] = predict_fraud(processed_data)

        # Show the processed data with predictions
        st.write("Processed Data with Predictions:")
        st.dataframe(processed_data)

        # Visualize the results
        visualize_results(processed_data)

# Button to check individual transaction
if st.button("Check Individual Transaction"):
    # Prepare data for prediction
    transaction_data = pd.DataFrame({
        "Date": [transaction_date],
        "Transaction_Type": [transaction_type],
        "Payment_Gateway": [payment_gateway],
        "Transaction_State": [transaction_state],
        "Merchant_Category": [merchant_category],
        "amount": [transaction_amount]
    })

    # Make prediction
    prediction = predict_fraud(transaction_data)
    if prediction[0] == 1:
        st.error("This transaction is likely to be fraudulent.")
    else:
        st.success("This transaction is likely to be legitimate.")

# Add custom CSS for styling
st.markdown(
    """
    <style>
    .reportview-container {
        background: url('https://cdn.sanity.io/images/9sed75bn/production/470934de877c88a13171081ae22e98994ce9cbd7-1792x1008.png') no-repeat center center fixed; 
        background-size: cover;
    }
    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.8);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Print XGBoost version (for debugging)
print(xgboost.__version__)
