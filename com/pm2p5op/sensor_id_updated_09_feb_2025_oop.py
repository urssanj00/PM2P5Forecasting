from AdvancedTemporalGraphNetwork import AdvancedTemporalGraphNetwork

from logger_config import logger
from PropertiesConfig import PropertiesConfig as PC
import pandas as pd
"""Process"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
# Importing
import sklearn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import torch
from torch import nn, optim
import time
class ProcessPM2P5Prediction:
    def __init__(self):  # 1
        logger.info("This is an info message.")

        properties_config = PC()
        self.properties = properties_config.get_properties_config()
        self.plot_path = self.properties['plot_path']
        self.input_path = self.properties['data_set_path']
        self.output_dir = self.properties['output_path']
        self.future_predictions_15min = self.properties['forecast_path']
        self.best_model_path = self.properties['best_model_path']

        logger.info(f'Drive Mounted - Input Path assigned {self.input_path}')
        logger.info(f'Drive Mounted - Best Model Path assigned {self.best_model_path}')
        logger.info(f'Drive Mounted - Plot Path assigned {self.plot_path}')

        self.combined_df = None
        self.dir_list = None # Dataset: list of files in input directory
        self.df = None

        self.sensor_ids = None

        self.X_train_tensor = None
        self.X_test_tensor = None
        self.y_train_tensor = None
        self.y_test_tensor = None

        self.scalers_X = {}
        self.scalers_y = {}

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.n_steps = 5

        # Get the list of all files and directories
    def list_files_of_a_dir(self, folder_name): # 2
        dir_path = f'{folder_name}'
        self.dir_list = os.listdir(dir_path)
        return self.dir_list

    def load_all_csv(self):  # 3 Load all CSV in a combined_df
        input_dir = f'{self.input_path}'
        logger.info(f'input_dir:{input_dir}')
        list_file = self.list_files_of_a_dir(input_dir)
        dataframes = []
        logger.info("Step 02: Loading selected columns from all CSVs into DataFrames")
        selected_columns = ['sensor_id', 'timestamp', 'temperature', 'humidity', 'longitude', 'latitude', 'pm2p5']
        i = 0
        for f_name in list_file:
            if f_name.endswith(".csv") and "station" not in f_name.lower():
                # Build the full file path
                csv_path = os.path.join(input_dir, f_name)

                try:
                    # Read the CSV file into a DataFrame with selected columns
                    df = pd.read_csv(csv_path, usecols=selected_columns)
                    dataframes.append(df)
                    logger.info(f"    {i}.    Loaded: {csv_path}")
                    i += 1
                except Exception as e:
                    logger.info(f"Error reading {csv_path}: {e}")

        if dataframes:
            self.combined_df = pd.concat(dataframes, ignore_index=True)
            self.combined_df = self.combined_df[selected_columns]  # Ensure columns are in the specified order
            logger.info(self.combined_df.columns.tolist())
            logger.info(f"Step 03: Clubbed selected columns from all CSVs into a dataset. Dataset Shape: ")
            logger.info(f"{self.combined_df.shape}")
        else:
            logger.info("No valid CSV files found.")
            self.combined_df = None

        return self.combined_df

    def plot_histogram(self):  # 4 Plot histogram to flag outliers
        self.df = self.load_all_csv()
        logger.info(self.df)
        plt.hist(self.df['pm2p5'], bins=20, color='blue')
        plt.xlabel('PM2.5')
        plt.ylabel('Frequency')
        plt.title('Histogram of PM2.5')
        plt.savefig(f'{self.plot_path}/histogram_before_outliers_plot.png')
        #plt.show()
        plt.close()

    def find_outliers_iqr(self, data, column):  # 5 Get the outliers using boxplot
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        logger.info(f'IQR : {IQR}')
        lower_bound = Q1 - 1.5 * IQR
        logger.info(f'lower_bound : {lower_bound}')
        upper_bound = Q3 + 1.5 * IQR
        logger.info(f'upper_bound : {upper_bound}')
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        return outliers
        # Example usage:
        # outliers = self.find_outliers_iqr(self.df, 'pm2p5')
        # logger.info(outliers)

    def boxplot_df(self):  # 6 Draw Boxplot
        # Assuming your DataFrame is named 'df':
        sns.boxplot(x=self.df['pm2p5'])  # Use seaborn for a more visually appealing boxplot
        plt.xlabel('PM2.5')
        plt.title('Boxplot of PM2.5')
        plt.savefig(f'{self.plot_path}/boxplot_before_outliers_plot.png')
        #plt.show()
        plt.close()

    def display_boxplot_outliers(self):  # 7 Print outliers
        top_50_pm2p5 = self.df.sort_values(by=['pm2p5'], ascending=True).head(50000)['pm2p5']
        logger.info(top_50_pm2p5)
        top_50_pm2p5 = self.df.sort_values(by=['pm2p5'], ascending=False).head(15000)['pm2p5']
        logger.info(top_50_pm2p5)

    def trim_outliers(self): # 8 Remove Outliers
        # Create the dataframe
        df_pm2p5 = self.df.copy()
        df_pm2p5.head()
        logger.info(f"Old Shape: {df_pm2p5.shape}")

        ''' Detection '''
        # IQR
        # Calculate the upper and lower limits
        Q1 = df_pm2p5['pm2p5'].quantile(0.25)
        Q3 = df_pm2p5['pm2p5'].quantile(0.75)
        IQR = Q3 - Q1
        logger.info(f'IQR:{IQR}')
        lower = Q1 - 1.5*IQR
        upper = Q3 + 1.5*IQR

        # Create arrays of Boolean values indicating the outlier rows
        upper_array = np.where(df_pm2p5['pm2p5'] >= upper)[0]
        lower_array = np.where(df_pm2p5['pm2p5'] <= lower)[0]

        # Removing the outliers
        df_pm2p5.drop(index=upper_array, inplace=True)
        df_pm2p5.drop(index=lower_array, inplace=True)
        df_pm2p5 = df_pm2p5[df_pm2p5['pm2p5'] > 0]

        # Print the new shape of the DataFrame
        logger.info(f"New Shape: {df_pm2p5.shape}")
        self.df = df_pm2p5

    def plot_boxplot_trimmed_df(self): # 8 BoxPlot Trimmed DF
        df_pm2p5 = self.df
        # Assuming your DataFrame is named 'df':
        sns.boxplot(x=df_pm2p5['pm2p5'])  # Use seaborn for a more visually appealing boxplot
        plt.xlabel('PM2.5')
        plt.title('Boxplot of PM2.5')
        plt.savefig(f'{self.plot_path}/boxplot_after_outliers_plot.png')
       # plt.show()
        plt.close()

    def plot_histogram_trimmed_df(self): # 9 Histogram Trimmed DF
        df_pm2p5 = self.df
        plt.hist(df_pm2p5['pm2p5'], bins=20, color='blue')
        plt.xlabel('PM2.5')
        plt.ylabel('Frequency')
        plt.title('Histogram of PM2.5')
        plt.savefig(f'{self.plot_path}/histogram_after_outliers_plot.png')
       # plt.show()
        plt.close()

    def display_trimmed_df(self):  # 11 Print trimmed
        df_pm2p5 = self.df
        top_50_pm2p5 = df_pm2p5.sort_values(by=['pm2p5'], ascending=True).head(50000)['pm2p5']
        logger.info(top_50_pm2p5)

        top_50_pm2p5 = df_pm2p5.sort_values(by=['pm2p5'], ascending=False).head(15000)['pm2p5']
        logger.info(top_50_pm2p5)

    def add_time_in_df(self):  # 12 split timestamp
        logger.info(self.df.columns)
        #pd.read_csv(data)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])

        self.df['year'] = self.df['timestamp'].dt.year
        self.df['month'] = self.df['timestamp'].dt.month
        self.df['day_of_month'] = self.df['timestamp'].dt.day
        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df['minute'] = self.df['timestamp'].dt.minute
        self.df['second'] = self.df['timestamp'].dt.second

        self.df = self.df.drop(columns=['timestamp'])

        columns = list(self.df.columns)
        columns.remove('pm2p5')
        #columns.remove('_id')
        columns.append('pm2p5')
        self.df = self.df[columns]
        #
    def scale_features(self, sensor_id):   # 13 scale features Create sequences and scale data by sensor_id
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        df_sensor = self.df[self.df['sensor_id'] == sensor_id]
        logger.info(f'scale_features->df_sensor==>')
        logger.info(f'{df_sensor}')
        logger.info(f'<==')
        X = df_sensor.drop(columns=['sensor_id', 'pm2p5']).to_numpy()
        y = df_sensor['pm2p5'].to_numpy()

        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        self.scalers_X[sensor_id] = scaler_X
        self.scalers_y[sensor_id] = scaler_y

        return X_scaled, y_scaled

    def create_sequences(self, X, y, n_steps): # 14 create Sequence
        Xs, ys = [], []
        for i in range(len(X) - n_steps + 1):
            Xs.append(X[i:i+n_steps])
            ys.append(y[i+n_steps-1])
        return np.array(Xs), np.array(ys)

    def create_scaled_tensor(self):
        # Define constants

      #  scalers_X, scalers_y = {}, {}
        self.sensor_ids = self.df['sensor_id'].unique()
        logger.info(f'sensor_ids : {self.sensor_ids}')
        # Scale and sequence the data
        X_seq_combined, y_seq_combined = [], []
        logger.info(f'before for loop sensor_ids:{self.sensor_ids}')

        for sensor_id in self.sensor_ids:
            logger.info(f'in for loop sensor_id:{sensor_id}')
            X_scaled, y_scaled = self.scale_features(sensor_id)
            X_seq, y_seq = self.create_sequences(X_scaled, y_scaled, self.n_steps)
            X_seq_combined.append(X_seq)
            y_seq_combined.append(y_seq)

        logger.info(f'0. Length X_seq_combined:{len(X_seq_combined)}')
        logger.info(f'0. Length y_seq_combined:{len(y_seq_combined)}')

        X_seq_combined = np.concatenate(X_seq_combined, axis=0)
        y_seq_combined = np.concatenate(y_seq_combined, axis=0)

        logger.info('Flattening the 2 lists X_seq_combined, y_seq_combined')
        logger.info(f'1. Length X_seq_combined:{len(X_seq_combined)}')
        logger.info(f'1. Length y_seq_combined:{len(y_seq_combined)}')

        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_seq_combined, y_seq_combined,
                                                            test_size=0.2, random_state=42)
        # Convert to tensors
        X_train_tensor = torch.tensor(self.X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(self.y_train, dtype=torch.float32)
        y_test_tensor = torch.tensor(self.y_test, dtype=torch.float32)
        return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

    def create_distance_based_edge_index(self, threshold_km=5.0):
        """
        Create edge index based on geographical distance between sensors
        Args:
            df: DataFrame containing 'sensor_id', 'latitude', 'longitude'
            threshold_km: Maximum distance in kilometers for creating an edge
        Returns:
            edge_index: torch tensor of shape [2, num_edges]
        """
        def haversine_distance(lat1, lon1, lat2, lon2):
            """Calculate the great circle distance between two points on Earth"""
            R = 6371  # Earth's radius in kilometers
            logger.info(f"Latitude in haversine_distance {lat1} {lon1} {lat2} {lon2}")
            # Convert decimal degrees to radians
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

            # Haversine formula
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            distance = R * c

            return distance

        # Get unique sensor locations
        sensor_locations = self.df.groupby('sensor_id')[['latitude', 'longitude']].mean().reset_index()
        num_sensors = len(sensor_locations)

        # Initialize lists to store edges
        edges_source = []
        edges_target = []

        # Create edges based on distance threshold
        for i in range(num_sensors):
            for j in range(i + 1, num_sensors):  # Only compute upper triangle to avoid duplicates
                logger.info(f"threshold_km {threshold_km}")
                distance = haversine_distance(
                    sensor_locations.iloc[i]['latitude'],
                    sensor_locations.iloc[i]['longitude'],
                    sensor_locations.iloc[j]['latitude'],
                    sensor_locations.iloc[j]['longitude']
                )

                logger.info(f"distance {distance} ")
                logger.info(f"distance {type(distance)} threshold_km {type(threshold_km)} ")
                if distance <= threshold_km:
                    # Add edges in both directions for undirected graph
                    edges_source.extend([i, j])
                    edges_target.extend([j, i])

        # If no edges were created based on distance, create a minimum spanning tree
        if not edges_source:
            logger.info("Warning: No edges created based on distance threshold. Creating minimum connectivity...")
            for i in range(num_sensors - 1):
                edges_source.extend([i, i+1])
                edges_target.extend([i+1, i])

        # Create self-loops for each node
        for i in range(num_sensors):
            edges_source.append(i)
            edges_target.append(i)

        # Convert to tensor
        edge_index = torch.tensor([edges_source, edges_target], dtype=torch.long)

        # Ensure the edge_index is on the same device as the model
        if torch.cuda.is_available():
            edge_index = edge_index.cuda()

        logger.info(f"Created edge_index with shape: {edge_index.shape}")
        logger.info(f"Number of nodes: {num_sensors}")
        logger.info(f"Number of edges: {edge_index.shape[1]}")

        return edge_index

    def train_and_test(self):
        # Usage in the training code:
        edge_index = self.create_distance_based_edge_index()

        self.X_train_tensor, self.X_test_tensor, self.y_train_tensor, self.y_test_tensor = (
            self.create_scaled_tensor())

        logger.info(self.X_train_tensor.shape)
        # Define and train the model
        num_features = self.X_train_tensor.shape[2]
        num_nodes = self.X_train_tensor.shape[1]

        model = AdvancedTemporalGraphNetwork(num_features=num_features,
                                             hidden_channels=16,
                                             num_nodes=num_nodes,
                                             dropout=0.2)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        num_epochs = int(self.properties['num_epochs'])
        train_losses, train_rmse_values, train_r2_values, train_mape_values = [], [], [], []
        test_losses, test_rmse_values, test_r2_values, test_mape_values = [], [], [], []

        best_train_predictions, best_test_predictions = None, None
        best_train_loss, best_test_loss = float('inf'), float('inf')

        logger.info(f'best_model_path {self.best_model_path}')

        for epoch in range(num_epochs):
            start_time = time.time()  # Record start time
           # logger.info(f"Epoch Start {start_time}")
            model.train()
            optimizer.zero_grad()
            predictions_train = model(self.X_train_tensor, edge_index, self.X_train_tensor)
            train_loss = criterion(predictions_train.squeeze(), self.y_train_tensor)
            train_loss.backward()
            optimizer.step()
            train_losses.append(train_loss.item())

            model.eval()
            with torch.no_grad():
                predictions_test = model(self.X_test_tensor, edge_index, self.X_test_tensor)
            test_loss = criterion(predictions_test.squeeze(), self.y_test_tensor)
            test_losses.append(test_loss.item())

            # Store the best predictions
            if test_loss.item() < best_test_loss:
                best_test_loss = test_loss.item()
                best_train_predictions = predictions_train.clone()
                best_test_predictions = predictions_test.clone()
                torch.save(model.state_dict(), self.best_model_path)  # Save the best model

            # Calculate metrics for training set
            predictions_train_unscaled = np.concatenate([self.scalers_y[sensor_id].inverse_transform(best_train_predictions[idx:idx+len(df[df['sensor_id'] == sensor_id])-self.n_steps+1].detach().numpy().reshape(-1, 1)).flatten() for idx, sensor_id in enumerate(self.sensor_ids)], axis=0)
            y_train_unscaled = np.concatenate([self.scalers_y[sensor_id].inverse_transform(self.y_train[idx:idx+len(df[df['sensor_id'] == sensor_id])-self.n_steps+1].reshape(-1, 1)).flatten() for idx, sensor_id in enumerate(self.sensor_ids)], axis=0)

            # Calculate metrics for testing set
            predictions_test_unscaled = np.concatenate([self.scalers_y[sensor_id].inverse_transform(best_test_predictions[idx:idx+len(df[df['sensor_id'] == sensor_id])-self.n_steps+1].detach().numpy().reshape(-1, 1)).flatten() for idx, sensor_id in enumerate(self.sensor_ids)], axis=0)
            y_test_unscaled = np.concatenate([self.scalers_y[sensor_id].inverse_transform(self.y_test[idx:idx+len(df[df['sensor_id'] == sensor_id])-self.n_steps+1].reshape(-1, 1)).flatten() for idx, sensor_id in enumerate(self.sensor_ids)], axis=0)
           #logger.info(f"{y_test_unscaled}, {predictions_test_unscaled}")
            train_rmse = np.sqrt(mean_squared_error(y_train_unscaled, predictions_train_unscaled))
            train_r2 = r2_score(y_train_unscaled, predictions_train_unscaled)
            train_mape = mean_absolute_percentage_error(y_train_unscaled, predictions_train_unscaled)

            test_rmse = np.sqrt(mean_squared_error(y_test_unscaled, predictions_test_unscaled))
            test_r2 = r2_score(y_test_unscaled, predictions_test_unscaled)
            test_mape = mean_absolute_percentage_error(y_test_unscaled, predictions_test_unscaled)

            train_rmse_values.append(train_rmse)
            train_r2_values.append(train_r2)
            train_mape_values.append(train_mape)
            test_rmse_values.append(test_rmse)
            test_r2_values.append(test_r2)
            test_mape_values.append(test_mape)
            end_time = time.time()  # Record end time
            epoch_time = end_time - start_time  # Calculate epoch time
            end_time = time.time()
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Time: {epoch_time:.2f} seconds, Train Loss: {train_loss.item()}, Test Loss: {test_loss.item()}, Train RMSE: {train_rmse}, Train R-Square: {train_r2}, Train MAPE: {train_mape}, Test RMSE: {test_rmse}, Test R-Square: {test_r2}, Test MAPE: {test_mape}")
          #  logger.info(f"Epoch End {end_time}")
            self.plot_train_and_test(train_losses, test_losses, train_rmse_values, test_rmse_values, train_r2_values,
                                     test_r2_values, train_mape_values, test_mape_values)

    def plot_train_and_test(self, train_losses, test_losses, train_rmse_values, test_rmse_values, train_r2_values,
                            test_r2_values, train_mape_values, test_mape_values):
        # Plotting train and test losses over epochs
        plt.figure(figsize=(12, 6))

        plt.subplot(2, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Train and Test Losses')
        plt.legend()

        # Plotting RMSE values over epochs
        plt.subplot(2, 2, 2)
        plt.plot(train_rmse_values, label='Train RMSE')
        plt.plot(test_rmse_values, label='Test RMSE')
        plt.xlabel('Epochs')
        plt.ylabel('RMSE')
        plt.title('Train and Test RMSE')
        plt.legend()

        # Plotting R² values over epochs
        plt.subplot(2, 2, 3)
        plt.plot(train_r2_values, label='Train R²')
        plt.plot(test_r2_values, label='Test R²')
        plt.xlabel('Epochs')
        plt.ylabel('R²')
        plt.title('Train and Test R²')
        plt.legend()

        # Plotting MAPE values over epochs
        plt.subplot(2, 2, 4)
        plt.plot(train_mape_values, label='Train MAPE')
        plt.plot(test_mape_values, label='Test MAPE')
        plt.xlabel('Epochs')
        plt.ylabel('MAPE')
        plt.title('Train and Test MAPE')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'{self.plot_path}/epoch_op_plot.png')
        #plt.show()
        plt.close()

    def prepare_future_data(self, future_dates, sensor_location_map):

        self.X_train_tensor, self.X_test_tensor, self.y_train_tensor, self.y_test_tensor = (
            self.create_scaled_tensor())

        """
        Prepare future data for forecasting with actual temperature & humidity.
        Args:
            future_dates: List of future dates (datetime objects).
            sensor_ids: List of sensor IDs.
            scalers_X: Dictionary of scalers for each sensor.
            sensor_location_map: Dictionary mapping sensor_id to (longitude, latitude).
            df: DataFrame containing historical data.
        Returns:
            X_future_tensor: Tensor of future data for prediction.
        """
        # Create DataFrame for future timestamps and sensor IDs
        logger.info(f"timestamp==> ")
        logger.info(f"{np.repeat(future_dates, len(self.sensor_ids))}")

        logger.info(f"sensor_id==> ")
        logger.info(f"{np.tile(self.sensor_ids, len(future_dates))}")

        future_df = pd.DataFrame({
            'timestamp': np.repeat(future_dates, len(self.sensor_ids)),
            'sensor_id': np.tile(self.sensor_ids, len(future_dates))
        })
        logger.info(f"0 future_df info")
        logger.info(f"0 {future_df.info}")
        logger.info(f"0 future_df info ends==>")

        # Get latest temperature & humidity readings per sensor
        latest_readings = self.df.groupby('sensor_id')[['temperature', 'humidity']].last()

        # Map temperature & humidity from historical data
        future_df['temperature'] = future_df['sensor_id'].map(lambda x: latest_readings.loc[x, 'temperature'] if x in latest_readings.index else np.nan)
        future_df['humidity'] = future_df['sensor_id'].map(lambda x: latest_readings.loc[x, 'humidity'] if x in latest_readings.index else np.nan)

        # Fill missing values with median (or any strategy)
        future_df['temperature'].fillna(future_df['temperature'].median(), inplace=True)
        future_df['humidity'].fillna(future_df['humidity'].median(), inplace=True)

        # Map longitude and latitude from sensor_location_map
        future_df['longitude'] = future_df['sensor_id'].map(lambda x: sensor_location_map[x]['longitude'] if x in sensor_location_map else None)
        future_df['latitude'] = future_df['sensor_id'].map(lambda x: sensor_location_map[x]['latitude'] if x in sensor_location_map else None)

        # Extract time-based features
        future_df['year'] = future_df['timestamp'].dt.year
        future_df['month'] = future_df['timestamp'].dt.month
        future_df['day_of_month'] = future_df['timestamp'].dt.day
        future_df['hour'] = future_df['timestamp'].dt.hour
        future_df['minute'] = future_df['timestamp'].dt.minute
        future_df['second'] = future_df['timestamp'].dt.second

        # Drop timestamp column
        future_df = future_df.drop(columns=['timestamp'])
        logger.info(f"1 future_df info")
        logger.info(f"1 {future_df.info}")
        logger.info(f"1 future_df info ends==>")
        # Scale features using the same scalers used during training
        X_future_scaled = []
        for sensor_id in self.sensor_ids:
            df_sensor = future_df[future_df['sensor_id'] == sensor_id].drop(columns=['sensor_id'])
            X_scaled = self.scalers_X[sensor_id].transform(df_sensor.to_numpy())
            X_future_scaled.append(X_scaled)

        # Combine scaled data
        X_future_scaled = np.concatenate(X_future_scaled, axis=0)
        logger.info(f"1 X_future_scaled info")
        logger.info(f"1 {X_future_scaled.shape}")
        logger.info(f"1 {X_future_scaled}")
        logger.info(f"1 X_future_scaled info ends==>")
        num_features = self.X_train_tensor.shape[2]
        num_nodes = self.X_train_tensor.shape[1]

        # Ensure divisibility by num_nodes (padding if necessary)
        total_samples = X_future_scaled.shape[0]
        remainder = total_samples % num_nodes
        if remainder != 0:
            num_dummy_rows = num_nodes - remainder
            dummy_rows = np.zeros((num_dummy_rows, num_features))  # Zero padding
            X_future_scaled = np.vstack([X_future_scaled, dummy_rows])

        X_future_scaled = X_future_scaled.reshape(-1, num_nodes, num_features)

        # Convert to tensor
        X_future_tensor = torch.tensor(X_future_scaled, dtype=torch.float32)
        logger.info(f"1 X_future_tensor info")
        logger.info(f"1 {X_future_tensor.shape}")
        logger.info(f"1 {X_future_tensor}")
        logger.info(f"1 X_future_scaled info ends==>")
        return X_future_tensor

    def future_dates_for_forcasting(self):
        # Define future dates for forecasting (every 15 minutes for the next 7 days)
        future_dates = pd.date_range(start=pd.Timestamp.now().floor('D') + pd.Timedelta(days=1),  # Start from next day
                                     end=pd.Timestamp.now().floor('D') + pd.Timedelta(days=8),    # End after 7 days
                                     freq='15min')  # Frequency: 15 minutes

        logger.info(f'len future_dates {len(future_dates)}')
        logger.info(f'{future_dates}')


        # Selecting relevant columns
        sensor_location_map = self.df[['sensor_id', 'longitude', 'latitude']]
        # Dropping duplicate sensor_id rows
        sensor_location_map = sensor_location_map.drop_duplicates(subset='sensor_id')

        # Setting index and converting to dictionary
        sensor_location_map = sensor_location_map.set_index('sensor_id')[['longitude', 'latitude']].to_dict(orient='index')
        logger.info(f"sensor_location_map : {sensor_location_map}")

        # Prepare future data
        X_future_tensor = self.prepare_future_data(future_dates, sensor_location_map)
        logger.info(f'X_future_tensor=>{X_future_tensor.shape}')
        logger.info(f'{X_future_tensor}')

        logger.info(f'len sensor_ids {len(self.sensor_ids)}')

        logger.info(f'best_model_path {self.best_model_path}')
        # Load the trained model
        num_features = self.X_train_tensor.shape[2]
        num_nodes = self.X_train_tensor.shape[1]
        model = AdvancedTemporalGraphNetwork(num_features=num_features,
                                             hidden_channels=16,
                                             num_nodes=num_nodes,
                                             dropout=0.2)
        # Usage in the training code:
        edge_index = self.create_distance_based_edge_index()

        model.load_state_dict(torch.load(self.best_model_path, weights_only=True))  # Load the best model
        model.eval()  # Set the model to evaluation mode
        # Generate predictions for future dates
        with torch.no_grad():
            future_predictions = model(X_future_tensor, edge_index, X_future_tensor)

        #Unscale the predictions
        future_predictions_unscaled = np.concatenate([
            self.scalers_y[sensor_id].inverse_transform(
                future_predictions[idx:idx+len(future_dates)].detach().numpy().reshape(-1, 1)).flatten()
            for idx, sensor_id in enumerate(self.sensor_ids)
        ], axis=0)

        # Validate final shape
        logger.info(f"  future_predictions_unscaled {len(future_predictions_unscaled)}")
        logger.info(f"  sensor_id {len(self.sensor_ids)}")
        logger.info(f"  timestamp {len(future_dates)}")

        expected_length = len(future_dates) * len(self.sensor_ids)
        logger.info(f"Expected length of predictions: {expected_length}")
        logger.info(f"Actual length of future_predictions_unscaled: {len(future_predictions_unscaled)}")

        if len(future_predictions_unscaled) < expected_length:
            logger.warning(
                f"Prediction length mismatch! Expected {expected_length}, got {len(future_predictions_unscaled)}.")
            future_predictions_unscaled = np.pad(future_predictions_unscaled,
                                                 (0, expected_length - len(future_predictions_unscaled)),
                                                 mode='constant')

        # Create a DataFrame to store the predictions
        future_predictions_df = pd.DataFrame({
            'timestamp': np.repeat(future_dates, len(self.sensor_ids)),
            'sensor_id': np.tile(self.sensor_ids, len(future_dates)),
            'predicted_pm2p5': future_predictions_unscaled
        })

        logger.info(self.future_predictions_15min)
        # Save the predictions to a CSV file
        # Drop rows where predicted_pm2p5 is 0
        future_predictions_df = future_predictions_df[future_predictions_df['predicted_pm2p5'] != 0]

        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.dirname(self.future_predictions_15min)
        logger.info(f"output_dir {output_dir}")
        for sensor_id, df_subset in future_predictions_df.groupby('sensor_id'):
            sensor_filename = f'future_predictions_sensor_{sensor_id}_{timestamp}.csv'
            df_subset.to_csv(f"{self.future_predictions_15min}/{sensor_filename}", index=False)
            logger.info(f"Saved predictions for sensor {sensor_id} to {sensor_filename}")
        logger.info("Future predictions saved to CSV.")

        logger.info(f"Future predictions saved to CSV.{self.future_predictions_15min}")

"""save with actual data"""
logger.info('Starting processing')
p = ProcessPM2P5Prediction()
logger.info(p)

list_of_dir = p.list_files_of_a_dir(p.input_path)
logger.info(list_of_dir)

df = p.load_all_csv()
logger.info(f'combined df {df}')
logger.info(f'plot_histogram')

p.plot_histogram()
logger.info(f'plot_histogram done')

outliers = p.find_outliers_iqr(df, 'pm2p5')
logger.info(f'Outliers')
logger.info(outliers)

p.boxplot_df()
logger.info(f'Saved boxplot with original dataset {p.df.shape}')

p.display_boxplot_outliers()

p.trim_outliers()
p.plot_boxplot_trimmed_df()

p.plot_histogram_trimmed_df()
logger.info(f'Saved histogram with trimmed dataset {p.df.shape}')

p.display_trimmed_df()

p.add_time_in_df()
logger.info(f'Added timestamp data in splits  {p.df.shape}')
logger.info(f"{p.df}")
logger.info(f"Printed modified dataframe")

logger.info(f"Creating Scaled Tensors")
p.train_and_test()
logger.info(f"Training completed and plots saved")
p.future_dates_for_forcasting()
logger.info(f"Future Date Forcast Done")
