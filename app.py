import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# Load and prepare data
def load_data(filepath):
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df


# Create sequences for LSTM input
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


# Build LSTM model
def build_model(seq_length, n_features):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(seq_length, n_features), return_sequences=True),
        LSTM(50, activation='relu'),
        Dense(n_features)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model


# Train model and make predictions
def train_and_predict(model, X_train, y_train, X_test, epochs=100, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    predictions = model.predict(X_test)
    return predictions


# Plot results
def plot_results(actual, predicted, product_name, dates):
    plt.figure(figsize=(14, 7))
    plt.plot(dates, actual, label='Actual', marker='o', markersize=3)
    plt.plot(dates, predicted, label='Predicted', marker='o', markersize=3)
    plt.title(f'Supply Demand Forecast for {product_name}')
    plt.xlabel('Date')
    plt.ylabel('Demand')
    plt.legend()

    # Format x-axis to show dates properly
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # Calculate the interval to ensure at least 5 ticks
    num_days = (dates[-1] - dates[0]).days
    interval = max(1, num_days // 5)  # Ensure interval is at least 1

    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    plt.gcf().autofmt_xdate()  # Rotate and align the tick labels

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

# Main function
def main():
    # Load data
    data = load_data('sales_data.csv')

    # Assuming the CSV has columns: Date, Product1, Product2, ...
    products = data.columns[1:]  # Exclude the 'Date' column

    for product in products:
        print(f"Forecasting for {product}")

        # Prepare data for the current product
        product_data = data[product].values.reshape(-1, 1)

        # Normalize data
        scaler = MinMaxScaler()
        product_data_scaled = scaler.fit_transform(product_data)

        # Create sequences
        seq_length = 10  # You can adjust this
        X, y = create_sequences(product_data_scaled, seq_length)

        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Build and train model
        model = build_model(seq_length, 1)
        predictions_scaled = train_and_predict(model, X_train, y_train, X_test)

        # Inverse transform predictions
        predictions = scaler.inverse_transform(predictions_scaled)

        # Get the corresponding dates for plotting
        plot_dates = data.index[train_size + seq_length:]

        # Plot results
        plot_results(product_data[train_size + seq_length:], predictions, product, plot_dates)


if __name__ == "__main__":
    main()
