import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN, Dropout  # Added Dropout
import streamlit as st

# Streamlit file upload interface
st.title("Stock Price Prediction with LSTM and RNN")
st.subheader("Upload your Excel file containing stock price data")

# File uploader widget
f = st.file_uploader("Choose an Excel file", type="xlsx")

# Check if a file is uploaded
if f is not None:
    # Load the data from the uploaded Excel file
    d = pd.read_excel(f)

    # Display the first few rows of the dataframe
    st.write(d.head())

    # Sort data by Date in ascending order (oldest to newest)
    d['D'] = pd.to_datetime(d['Date'])
    d = d.sort_values('D')

    # Calculate 20-day moving average and standard deviation
    d['M'] = d['Price'].rolling(window=20).mean()
    d['S'] = d['Price'].rolling(window=20).std()

    # Calculate upper and lower Bollinger Bands
    d['U'] = d['M'] + 2 * d['S']
    d['L'] = d['M'] - 2 * d['S']

    # Plot Bollinger Bands
    plt.figure(figsize=(14, 7))
    plt.plot(d['D'], d['Price'], label='Actual Price', color='blue')
    
    plt.plot(d['D'], d['U'], label='Upper Band', color='green', linestyle='--')
    plt.plot(d['D'], d['L'], label='Lower Band', color='red', linestyle='--')

    plt.title('Bollinger Bands on Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)  # Display the plot using Streamlit

    # Assuming the relevant column is named 'Price' for stock prices
    p = d['Price'].values.reshape(-1, 1)

    # Normalize the data using MinMaxScaler
    s = MinMaxScaler(feature_range=(0, 1))
    t = s.fit_transform(p)

    # Prepare the dataset for training (using the last 'n' days for prediction)
    def c(a, b=60):
        x, y = [], []
        for i in range(len(a) - b - 1):
            x.append(a[i:(i + b), 0])
            y.append(a[i + b, 0])
        return np.array(x), np.array(y)

    # Create the dataset
    b = 60
    x, y = c(t, b)
    x = x.reshape(x.shape[0], x.shape[1], 1)

    # Split into training and testing
    m = int(len(x) * 0.8)
    x_train, x_test = x[:m], x[m:]
    y_train, y_test = y[:m], y[m:]

    # Build Stacked LSTM model with Dropout (Updated)
    def l(i):
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=i))
        model.add(Dropout(0.2))
        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    # Build RNN model with Dropout (Updated)
    def r(i):
        model = Sequential()
        model.add(SimpleRNN(32, return_sequences=True, input_shape=i))
        model.add(Dropout(0.2))
        model.add(SimpleRNN(64, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(SimpleRNN(32, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    # Train both models (Epochs updated to 30)
    m_lstm = l((x_train.shape[1], 1))
    m_rnn = r((x_train.shape[1], 1))

    h_lstm = m_lstm.fit(x_train, y_train, epochs=30, batch_size=32, validation_data=(x_test, y_test), verbose=1)
    h_rnn = m_rnn.fit(x_train, y_train, epochs=30, batch_size=32, validation_data=(x_test, y_test), verbose=1)

    # Predict
    p_lstm_train = m_lstm.predict(x_train)
    p_lstm_test = m_lstm.predict(x_test)
    p_rnn_train = m_rnn.predict(x_train)
    p_rnn_test = m_rnn.predict(x_test)

    # Average ensemble
    p_train = (p_lstm_train + p_rnn_train) / 2
    p_test = (p_lstm_test + p_rnn_test) / 2

    # Inverse transform
    p_train = s.inverse_transform(p_train)
    p_test = s.inverse_transform(p_test)
    y_train_actual = s.inverse_transform(y_train.reshape(-1, 1))
    y_test_actual = s.inverse_transform(y_test.reshape(-1, 1))

    # Plot training predictions
    plt.figure(figsize=(14, 8))
    d_train = d['D'][:m]
    plt.plot(d_train, y_train_actual, color='blue', label='Actual Price (Train)')
    plt.plot(d_train, p_train, color='green', label='Ensemble Prediction (Train)')

    # Plot testing predictions
    d_test = d['D'][m:m + len(y_test_actual)]
    plt.plot(d_test, y_test_actual, color='red', label='Actual Price (Test)')
    plt.plot(d_test, p_test, color='green', linestyle='--', label='Ensemble Prediction (Test)')

    # Next day prediction
    n_input = x_test[-1].reshape(1, b, 1)
    n_lstm = m_lstm.predict(n_input)
    n_rnn = m_rnn.predict(n_input)
    n_pred = (n_lstm + n_rnn) / 2
    n_pred = s.inverse_transform(n_pred)

    n_date = d_test.iloc[-1]
    plt.scatter(n_date, n_pred, color='yellow', label='Next Day Prediction', zorder=5)

    plt.title('Stock Price Prediction (TCS) - Ensemble of Stacked LSTM and RNN')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    st.pyplot(plt)  # Display the plot using Streamlit

    # Evaluate performance
    def e(a, b, name):
        mse = mean_squared_error(a, b)
        mape = mean_absolute_percentage_error(a, b)
        accuracy = 100 - mape * 100
        st.write(f"{name} => MSE: {mse:.4f}, MAPE: {mape:.4f}, Accuracy: {accuracy:.2f}%")

    e(y_train_actual, p_train, "Ensemble Train")
    e(y_test_actual, p_test, "Ensemble Test")

    # Evaluate LSTM and RNN individually (New)
    p_lstm_test_actual = s.inverse_transform(p_lstm_test)
    p_rnn_test_actual = s.inverse_transform(p_rnn_test)
    e(y_test_actual, p_lstm_test_actual, "LSTM Test")
    e(y_test_actual, p_rnn_test_actual, "RNN Test")

    # Plot Training and Validation Losses
    plt.figure(figsize=(10, 6))
    plt.plot(h_lstm.history['loss'], label='LSTM Training Loss', color='blue')
    plt.plot(h_lstm.history['val_loss'], label='LSTM Validation Loss', color='red')
    plt.plot(h_rnn.history['loss'], label='RNN Training Loss', color='purple')
    plt.plot(h_rnn.history['val_loss'], label='RNN Validation Loss', color='orange')
    plt.title('Training and Validation Loss (LSTM + RNN)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)  # Display the plot using Streamlit

    # Comparison Plot: Actual vs RNN, LSTM, and Ensemble Predictions (Normalized)
    lstm_test_pred_norm = p_lstm_test.flatten()
    rnn_test_pred_norm = p_rnn_test.flatten()
    ensemble_test_pred_norm = ((p_lstm_test + p_rnn_test) / 2).flatten()
    y_test_norm = y_test.flatten()

    plt.figure(figsize=(14, 7))
    plt.plot(y_test_norm, label='Actual Prices', color='blue')
    plt.plot(rnn_test_pred_norm, label='RNN Predictions', color='green', linestyle='dashed')
    plt.plot(lstm_test_pred_norm, label='LSTM Predictions', color='orange')
    plt.plot(ensemble_test_pred_norm, label='Ensemble Predictions', color='red')

    plt.title('Comparison of Actual Prices vs RNN, LSTM, and Ensemble Predictions')
    plt.xlabel('Test Samples')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)  # Display the plot using Streamlit
