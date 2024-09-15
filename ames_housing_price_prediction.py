import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import xgboost as xgb

class AmesHousingModel:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data = None
        self.df_encoded = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

    def load_and_prepare_data(self):
        self.data = pd.read_csv(self.csv_path)

        #Feature engineering: TotalArea and PriceperArea
        self.data['TotalArea'] = (
            self.data['Lot Area'] + self.data['Mas Vnr Area'] + self.data['Gr Liv Area'] +
            self.data['Garage Area'] + self.data['Pool Area']
        )
        self.data['PriceperArea'] = self.data['SalePrice'] / self.data['TotalArea']

        #One-hot encoding non-numeric data
        non_numeric_cols = self.data.select_dtypes(exclude=['number']).columns
        self.df_encoded = pd.get_dummies(self.data, columns=non_numeric_cols, drop_first=True)

        #Fix boolean columns
        bool_cols = self.df_encoded.select_dtypes(include='bool').columns
        self.df_encoded[bool_cols] = self.df_encoded[bool_cols].astype(int)

        #Scaling the data
        numeric_cols = self.df_encoded.select_dtypes(include=['number']).columns
        scaler = StandardScaler()
        self.df_encoded[numeric_cols] = scaler.fit_transform(self.df_encoded[numeric_cols])

        #Handling NaN values
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        X_train_imputed = imputer.fit_transform(self.df_encoded.drop('SalePrice', axis=1))
        self.X_train = pd.DataFrame(X_train_imputed, columns=self.df_encoded.drop('SalePrice', axis=1).columns)
        self.y_train = self.data['SalePrice']

    def split_data(self):
        split_1 = int(0.6 * len(self.X_train))
        split_2 = int(0.8 * len(self.X_train))

        self.X_train, self.X_val, self.X_test = (
            self.X_train[:split_1], self.X_train[split_1:split_2], self.X_train[split_2:]
        )
        self.y_train, self.y_val, self.y_test = (
            self.y_train[:split_1], self.y_train[split_1:split_2], self.y_train[split_2:]
        )

    def plot_insights(self):
        plt.scatter(self.data['TotalArea'], self.data['SalePrice'])
        plt.xlabel('Area')
        plt.ylabel('Sale Price')
        plt.show()

    def train_neural_network(self, epochs=100, batch_size=100):
        model = Sequential()
        model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01), input_shape=(self.X_train.shape[1],)))
        model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dense(1, activation='linear'))

        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mse'])
        model.fit(self.X_train, self.y_train, epochs=epochs, validation_data=(self.X_val, self.y_val), batch_size=batch_size)

        train_pred = model.predict(self.X_train)
        val_pred = model.predict(self.X_test)
        return self.evaluate_model(self.y_train, train_pred, self.y_test, val_pred)

    def train_random_forest(self, n_estimators=100):
        model = RandomForestRegressor(n_estimators=n_estimators)
        model.fit(self.X_train, self.y_train)

        train_pred = model.predict(self.X_train)
        val_pred = model.predict(self.X_test)
        return self.evaluate_model(self.y_train, train_pred, self.y_test, val_pred)

    def train_gradient_boosting(self, n_estimators=1000):
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=n_estimators)
        model.fit(self.X_train, self.y_train)

        train_pred = model.predict(self.X_train)
        val_pred = model.predict(self.X_test)
        return self.evaluate_model(self.y_train, train_pred, self.y_test, val_pred)

    @staticmethod
    def evaluate_model(y_train, y_train_pred, y_test, y_test_pred):
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        print(f"Train MSE: {train_mse}")
        print(f"Test MSE: {test_mse}")

        return test_mse

    def evaluate_performance(self, test_mse):
        rmse = np.sqrt(test_mse)
        average_price = self.data['SalePrice'].mean()
        rmse_percentage = (rmse / average_price) * 100

        def categorize_rmse_percentage(rmse_percentage):
            if rmse_percentage <= 5:
                return "Perfect"
            elif 5 < rmse_percentage <= 10:
                return "Great"
            elif 10 < rmse_percentage <= 15:
                return "Good"
            elif 15 < rmse_percentage <= 20:
                return "Fair"
            else:
                return "Needs Improvement"

        print(f"RMSE as a percentage of average house price: {rmse_percentage}% , considered {categorize_rmse_percentage(rmse_percentage)}")

    def run(self):
        self.load_and_prepare_data()
        self.split_data()
        self.plot_insights()

        # Neural Network
        print("Neural Network Performance:")
        nn_mse = self.train_neural_network()

        # Random Forest
        print("\nRandom Forest Performance:")
        rf_mse = self.train_random_forest()

        # Gradient Boosting
        print("\nGradient Boosting Performance:")
        gb_mse = self.train_gradient_boosting()

        # Evaluating best performance
        best_mse = min(nn_mse, rf_mse, gb_mse)
        self.evaluate_performance(best_mse)

# For running in colab
from google.colab import drive
drive.mount('/content/drive')
csv_path = '/content/drive/MyDrive/ML Datasets/AmesHousing.csv'
housing_model = AmesHousingModel(csv_path)
housing_model.run()