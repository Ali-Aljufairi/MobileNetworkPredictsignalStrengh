import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


class MobileNetwork:
    def __init__(self, num_points, frequency, height_transmitter, height_receiver, terrain):
        self.num_points = num_points
        self.frequency = frequency
        self.height_transmitter = height_transmitter
        self.height_receiver = height_receiver
        self.terrain = terrain

        self._generate_data()

    def _generate_data(self):
        d = np.random.uniform(0.1, 10, self.num_points)  # distance in km

        terrain_correction = {'urban': 0, 'suburban': -2, 'rural': -5}
        C = terrain_correction.get(self.terrain, 0)

        A_h = lambda h: (1.1 * np.log10(self.frequency) - 0.7) * h - (1.56 * np.log10(self.frequency) - 0.8)
        A_v = lambda h: 8.29 * (np.log10(1.54 * h)) ** 2 - 1.1
        A = A_h(self.height_receiver) + A_v(self.height_receiver) - 2.7 * np.log10(self.height_receiver / 3)
        L = lambda d: 69.55 + 26.16 * np.log10(self.frequency) - 13.82 * np.log10(self.height_transmitter) - A + (
                    44.9 - 6.55 * np.log10(self.height_transmitter)) * np.log10(d) + C
        path_loss = L(d) + np.random.randn(self.num_points) * 5  # add some random noise
        reference_signal = -50  # signal strength at reference distance (d0)
        signal_strengths = reference_signal - path_loss

        self.data = pd.DataFrame({'distance': d * 1000, 'signal_strength': signal_strengths})

    def split_data(self, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(self.data.drop('signal_strength', axis=1),
                                                            self.data['signal_strength'],
                                                            test_size=test_size,
                                                            random_state=random_state)
        return X_train, X_test, y_train, y_test


class SignalStrengthModel:
    def __init__(self, name, model):
        self.name = name
        self.model = model

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        print(f"{self.name} RMSE:", rmse)

    def predict(self, new_data):
        predictions = self.model.predict(new_data)
        print(f"{self.name} predictions:", predictions)


def main():
    # Set the parameters of the Okumura-Hata model
    num_points = 1000
    frequency = 900  # frequency in MHz
    height_transmitter = 80  # height of transmitter in meters
    height_receiver = 1.5  # height of receiver in meters
    terrain = 'urban'

    # Create a mobile network object and generate data
    mobile_network = MobileNetwork(num_points, frequency, height_transmitter, height_receiver, terrain)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = mobile_network.split_data()

    # Scale the input features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train signal strength models
    models = [
        SignalStrengthModel("Linear Regression", LinearRegression()),
        SignalStrengthModel("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42)),
        SignalStrengthModel("XGBoost", XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
    ]
    for model in models:
        model.train(X_train, y_train)
        model.evaluate(X_test, y_test)

    # Use the models to make predictions
    new_data = pd.DataFrame({'distance': [100, 200, 300]})
    new_data = scaler.transform(new_data)
    for model in models:
        model.predict(new_data)


if __name__ == "__main__":
    main()