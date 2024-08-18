import numpy as np

class ARIMA:
    def __init__(self, p, d, q):
        self.p = p  # Order of AR part
        self.d = d  # Order of differencing
        self.q = q  # Order of MA part
        self.ar_params = np.zeros(p)
        self.ma_params = np.zeros(q)
        self.mean = 0
    
    def fit(self, data):
        data = np.array(data)
        self.mean = np.mean(data)
        data_diff = self.difference(data, self.d)
        
        # Fit AR model
        if self.p > 0:
            self.ar_params = self.fit_ar_model(data_diff)
        
        # Fit MA model
        if self.q > 0:
            self.ma_params = self.fit_ma_model(data_diff)
    
    def difference(self, data, d):
        """Perform differencing on the data."""
        for _ in range(d):
            data = np.diff(data)
        return data
    
    def fit_ar_model(self, data):
        """Fit AutoRegressive model (AR part)."""
        n = len(data)
        if self.p > 0:
            X = np.vstack([data[i:n - self.p + i] for i in range(self.p)]).T
            y = data[self.p:]
            if X.shape[0] > 0:
                ar_params = np.linalg.pinv(X).dot(y)
            else:
                ar_params = np.zeros(self.p)
        else:
            ar_params = np.zeros(self.p)
        return ar_params
    
    def fit_ma_model(self, data):
        """Fit Moving Average model (MA part)."""
        n = len(data)
        if self.q > 0:
            residuals = np.zeros(n)
            errors = np.zeros(n)
            
            # Compute residuals from the AR part
            if self.p > 0:
                ar_part = np.convolve(data, np.flip(self.ar_params), 'valid')
            else:
                ar_part = np.zeros(n)
            
            residuals[self.p:] = data[self.p:] - ar_part[self.p-1:]
            
            # Fit MA model
            X = np.vstack([residuals[i:n - self.q + i] for i in range(self.q)]).T
            y = residuals[self.q:]
            if X.shape[0] > 0:
                ma_params = np.linalg.pinv(X).dot(y)
            else:
                ma_params = np.zeros(self.q)
        else:
            ma_params = np.zeros(self.q)
        return ma_params
    
    def predict(self, steps, data):
        """Forecast future values based on the AR and MA parameters."""
        forecast = np.zeros(steps)
        data = np.array(data)
        n = len(data)
        
        # Use the last values to start predictions
        current_data = data[-max(self.p, self.q):]
        
        for i in range(steps):
            # Calculate AR and MA contributions
            ar_part = np.sum(self.ar_params * current_data[-self.p:]) if self.p > 0 else 0
            ma_part = np.sum(self.ma_params * np.random.randn(self.q)) if self.q > 0 else 0
            
            forecast[i] = self.mean + ar_part + ma_part
            
            # Update current data
            current_data = np.append(current_data, forecast[i])[-max(self.p, self.q):]
        
        return forecast

# Example Usage
if __name__ == "__main__":
    # Sample time series data
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    # Create and fit the ARIMA model
    model = ARIMA(p=2, d=1, q=2)
    model.fit(data)
    
    # Predict future values
    forecast = model.predict(steps=5, data=data)
    print("Forecasted values:", forecast)
