import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Set up Streamlit layout and theme
st.set_page_config(layout="wide", page_title="Milk Production Analysis")
st.title("Monthly Milk Production Analysis")

# Upload the data
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, index_col='Month', parse_dates=True)
    data.rename(columns={"Monthly milk production (pounds per cow)": "Milk Production"}, inplace=True)

    # Display the data overview
    st.subheader("Dataset Overview")
    st.write(data.head())

    # Rename the column for consistency
    data.rename(columns={"Milk Production": "Production"}, inplace=True)

    # Add moving averages to the data
    data['SMA_12'] = data['Production'].rolling(window=12).mean()
    data['SMA_12_shifted'] = data['SMA_12'].shift(1)
    data['EMA_12'] = data['Production'].ewm(span=12, adjust=False).mean()
    data['EMA_12_shifted'] = data['EMA_12'].shift(1)
    data['Custom_EMA_0.6'] = data['Production'].ewm(alpha=0.6, adjust=False).mean()
    data['Custom_EMA_0.6_shifted'] = data['Custom_EMA_0.6'].shift(1)

    # Plot line chart with different moving averages
    st.subheader("Milk Production with Moving Averages")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['Production'], label='Production', color='black', linewidth=1.5)
    ax.plot(data['SMA_12'], label='12-Month SMA', color='orange')
    ax.plot(data['SMA_12_shifted'], label='12-Month SMA Shifted', color='red', linestyle='--')
    ax.plot(data['EMA_12'], label='12-Month EMA', color='green')
    ax.plot(data['EMA_12_shifted'], label='12-Month EMA Shifted', color='blue', linestyle='--')
    ax.plot(data['Custom_EMA_0.6'], label='Custom EMA (alpha=0.6)', color='purple')
    ax.plot(data['Custom_EMA_0.6_shifted'], label='Custom EMA Shifted (alpha=0.6)', color='purple', linestyle='--')
    ax.set_title('Monthly Milk Production with Moving Averages')
    ax.set_xlabel('Date')
    ax.set_ylabel('Milk Production (pounds)')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Display Histogram for Milk Production
    st.subheader("Histogram of Monthly Milk Production")
    fig, ax = plt.subplots()
    sns.histplot(data['Production'], bins=20, kde=True, color='skyblue', ax=ax)
    ax.set_title('Distribution of Monthly Milk Production')
    ax.set_xlabel('Milk Production (pounds)')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # Display Bar Chart for Monthly Milk Production (first 12 months)
    st.subheader("Bar Chart of Monthly Milk Production - First 12 Months")
    fig, ax = plt.subplots()
    data['Production'][:12].plot(kind='bar', color='coral', ax=ax)
    ax.set_title('Monthly Milk Production - First 12 Months')
    ax.set_xlabel('Month')
    ax.set_ylabel('Milk Production (pounds)')
    st.pyplot(fig)

    # Split the data into train and test sets
    train, test = train_test_split(data, test_size=0.2, shuffle=False)

    # Exponential Smoothing Forecast (Manual Implementation)
    alpha = 0.1  # Set the smoothing level
    train_production = train['Production'].values
    forecast = [train_production[0]]  # Initialize forecast list with the first production value

    # Calculate forecast for each period in the test set
    for i in range(1, len(train_production)):
        forecast_value = alpha * train_production[i-1] + (1 - alpha) * forecast[-1]
        forecast.append(forecast_value)

    # Extend forecast to cover test data period
    for _ in range(len(test)):
        forecast_value = alpha * forecast[-1] + (1 - alpha) * forecast[-1]
        forecast.append(forecast_value)

    # Prepare forecast DataFrame for comparison
    forecast_df = pd.DataFrame({'Actual': test['Production'], 'Forecast': forecast[-len(test):]}, index=test.index)

    # Display Actual vs Forecast Plot
    st.subheader("Actual vs Forecast Milk Production")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train['Production'], label='Training Data', color='black')
    ax.plot(test['Production'], label='Test Data', color='blue')
    ax.plot(test.index, forecast_df['Forecast'], label='Forecasted Data', color='orange')
    ax.set_title("Actual vs Forecast Milk Production")
    ax.set_xlabel('Date')
    ax.set_ylabel('Milk Production (pounds)')
    ax.legend()
    st.pyplot(fig)

    # Show Forecasted Data with Test Data
    st.subheader("Forecasted vs Actual Test Data")
    st.write(forecast_df.head())
    st.line_chart(forecast_df)

    # Additional Summary Metrics
    st.subheader("Summary Statistics")
    st.write(data.describe())

    # Customize the sidebar for additional controls or filtering
    st.sidebar.header("Filter Options")
    selected_columns = st.sidebar.multiselect("Select columns to display", data.columns, default=data.columns)
    filtered_data = data[selected_columns]

    st.sidebar.header("Choose Plot Colors")
    production_color = st.sidebar.color_picker("Production Line Color", "#000000")
    sma_color = st.sidebar.color_picker("SMA Line Color", "#FFA500")
    ema_color = st.sidebar.color_picker("EMA Line Color", "#008000")
    forecast_color = st.sidebar.color_picker("Forecast Line Color", "#FF6347")

    # Updated plot with user-selected colors
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['Production'], label='Production', color=production_color)
    ax.plot(data['SMA_12'], label='12-Month SMA', color=sma_color)
    ax.plot(data['EMA_12'], label='12-Month EMA', color=ema_color)
    ax.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', color=forecast_color)
    ax.set_title("Customized Plot with Selected Colors")
    ax.set_xlabel('Date')
    ax.set_ylabel('Milk Production (pounds)')
    ax.legend()
    st.pyplot(fig)
else:
    st.write("Please upload a CSV file to proceed.")
