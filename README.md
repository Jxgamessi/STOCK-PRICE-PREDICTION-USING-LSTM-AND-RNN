📝 Project Overview
This project implements a stock price prediction system using deep learning models (Stacked LSTM and Stacked Simple RNN) combined into an ensemble for improved forecasting.
It is built as an interactive Streamlit web application, allowing users to upload their own stock datasets in .xlsx format, visualize Bollinger Bands, train models, predict future prices, and analyze performance metrics.

🚀 Key Features
📄 Upload custom stock price datasets (Excel .xlsx files).

📊 Visualize Bollinger Bands for stock price volatility.

🧠 Train Stacked LSTM and Stacked Simple RNN models with Dropout regularization.

🧩 Perform ensemble averaging for better prediction accuracy.

📈 Predict next-day stock prices.

📉 Plot Training vs Validation Loss for both models.

📊 Display evaluation metrics: MSE, MAPE, and Accuracy.

🛠️ Technology Stack
>> Python 3.x

>> Streamlit – for creating the web app

>> Pandas, NumPy – for data handling

>> Matplotlib – for plotting and visualizations

>> Scikit-learn – for data scaling and performance metrics

>> Keras (TensorFlow backend) – for building LSTM and RNN models

>> Openpyxl – for reading .xlsx Excel


Install the required Python libraries:

pip install pandas numpy matplotlib scikit-learn keras streamlit openpyxl


