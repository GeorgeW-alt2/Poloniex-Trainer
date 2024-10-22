import requests
import numpy as np
import torch
import torch.nn as nn

# 1. Poloniex SDK to Fetch Data
class PoloniexSDK:
    def __init__(self):
        self.base_url = "https://api.poloniex.com"

    def get_trade_history(self, symbol, limit):
        path = f"/markets/{symbol}/trades"
        params = {"limit": limit}
        response = requests.get(self.base_url + path, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception("Error fetching trade history")

# 2. Data Preprocessing
def manual_min_max_scaler(data):
    min_value = np.min(data)
    max_value = np.max(data)
    scaled_data = (data - min_value) / (max_value - min_value)
    return scaled_data, min_value, max_value

def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length + 1])  # Include the target as the last element
    return np.array(sequences)

# 3. Define the Neural Network
class PricePredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PricePredictionModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Get the last time step
        return out

# 4. Training the Model
def train_model(model, X, y, epochs=100, batch_size=32):
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Convert to tensors
    X_tensor = torch.FloatTensor(X).view(-1, seq_length, input_size)
    y_tensor = torch.FloatTensor(y).view(-1, 1)

    for epoch in range(epochs):
        model.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# 5. Making Predictions
def predict(model, data, min_value, max_value, steps=5):
    model.eval()
    predictions = []
    current_data = torch.FloatTensor(data).view(-1, seq_length, input_size)  # Convert to tensor

    for _ in range(steps):
        with torch.no_grad():
            # Make the prediction
            predicted = model(current_data)
            predicted_price = predicted.numpy()[0, 0]  # Get the predicted value
            
            # Inverse scale the predicted price
            predicted_price_actual = predicted_price * (max_value - min_value) + min_value
            predictions.append(predicted_price_actual)  # Append the predicted price
            
            # Prepare new input for the next step
            new_input = torch.FloatTensor([predicted_price]).view(1, 1, input_size)  # Wrap in a list
            current_data = torch.cat((current_data[:, 1:, :], new_input), dim=1)  # Update the input sequence

    return np.array(predictions)

# Main Execution
if __name__ == "__main__":
    while True:
        # Fetch Data
        sdk = PoloniexSDK()
        prices = sdk.get_trade_history("BTC_USDT", 1000)

        # Preprocess Data
        closing_prices = np.array([float(price['price']) for price in prices])
        scaled_prices, min_value, max_value = manual_min_max_scaler(closing_prices)

        seq_length = 10
        sequences = create_sequences(scaled_prices, seq_length)
        X = sequences[:, :-1]  # All but last column
        y = sequences[:, -1]   # Last column (target)

        # Define Model
        input_size = 1  # Price at each time step
        hidden_size = 64
        model = PricePredictionModel(input_size, hidden_size)

        # Train Model
        train_model(model, X, y)
        
        # Display the last five prices
        print("Last five closing prices:")
        print(closing_prices[-5:])
        
        # Make Predictions for 5 steps into the future
        latest_data = scaled_prices[-seq_length:]  # Use the last seq_length prices
        predicted_prices = predict(model, latest_data, min_value, max_value, steps=5)
        
        print("Predicted prices for the next 5 steps:")
        for i, price in enumerate(predicted_prices):
            print(f'Step {i + 1}: {price:.4f}')

        # Determine if the price is going up or down
        last_actual_price = closing_prices[-1]
        predicted_price_value = predicted_prices[-1]

        if predicted_price_value > last_actual_price:
            print("The predicted price is going up.")
        else:
            print("The predicted price is going down.")
        input("Press enter to continue.")
        print()
