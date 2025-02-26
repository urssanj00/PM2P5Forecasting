import pandas as pd
from flask import Flask, render_template, request
from PropertiesConfig import PropertiesConfig as PC
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

properties_config = PC()
properties = properties_config.get_properties_config()
future_predictions_15min = properties['forecast_path']
# Read station data from CSV
# Read all CSV files from the given path
def load_station_data():
    forecast_path = properties['forecast_path']
    all_files = [f for f in os.listdir(forecast_path) if f.endswith('.csv')]  # List all CSV files
    df_list = [pd.read_csv(os.path.join(forecast_path, file)) for file in all_files]  # Read each CSV

    # Combine all data into a single DataFrame
    df = pd.concat(df_list, ignore_index=True)

    df["sensor_id"] = df["sensor_id"].astype(str)  # Convert sensor_id to string
    stations = df["sensor_id"].unique().tolist()  # Get unique station IDs

    return df, stations

df, stations = load_station_data()

@app.route("/", methods=["GET"])
def index():
    selected_station = request.args.get("station_id")  # Get station_id from URL
    selected_data = None

    if selected_station:
        selected_data = df[df["sensor_id"] == selected_station][["timestamp", "sensor_id", "predicted_pm2p5"]].values.tolist()

    return render_template("index.html", stations=stations, selected_station=selected_station, selected_data=selected_data)


if __name__ == "__main__":
    app.run(debug=True)
