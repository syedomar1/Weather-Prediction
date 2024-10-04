import warnings
warnings.filterwarnings('ignore')

# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from shapely.geometry import Point
from datetime import timedelta

# Read CSV file with weather data
weather_df = pd.read_csv('pune2.csv', parse_dates=['date_time'], index_col='date_time')

# Display the first few rows
print(weather_df.head())

# Fixing the column mismatch issue
# Check for tempC column existence
if 'tempC' not in weather_df.columns:
    raise KeyError("The 'tempC' column is missing from the dataset!")

# Drop columns that are not relevant for the model
weather_df = weather_df.drop(['moonrise', 'moonset', 'sunrise', 'sunset', 'totalSnow_cm'], axis=1)

# Handle missing data by forward filling
weather_df.fillna(method='ffill', inplace=True)

# One-Hot Encoding for the 'place' column
encoder = OneHotEncoder(sparse_output=False)
place_encoded = encoder.fit_transform(weather_df[['place']])

# Create a DataFrame for the encoded 'place' data and concatenate with the original DataFrame
place_encoded_df = pd.DataFrame(place_encoded, columns=encoder.get_feature_names_out(['place']), index=weather_df.index)
weather_df = pd.concat([weather_df, place_encoded_df], axis=1)

# Drop the original 'place' column (we have encoded it now)
weather_df.drop(['place'], axis=1, inplace=True)

# Data exploration: Checking the dataframe structure
print(weather_df.head())
print(weather_df.columns)
print(weather_df.describe())

# Selecting the target variable (tempC) and features
features = weather_df.drop(['tempC'], axis=1)
target = weather_df['tempC']  # Target is tempC (temperature)

# Save the updated file
weather_df.to_csv('pune_encoded.csv', index=False)

print("Categorical data encoded and saved to pune_encoded.csv")

# Splitting data into training and testing sets (80% train, 20% test)
train_X, test_X, train_y, test_y = train_test_split(features, target, test_size=0.2, random_state=42)

# Model training with RandomForestRegressor
regr = RandomForestRegressor(n_estimators=100, random_state=42)
regr.fit(train_X, train_y)

# Predict on the test set
test_predictions = regr.predict(test_X)

# Evaluating model performance
mae = mean_absolute_error(test_y, test_predictions)
mse = mean_squared_error(test_y, test_predictions)
r2 = r2_score(test_y, test_predictions)

print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'R^2 Score: {r2:.2f}')

# Showing actual vs predicted temperatures
result_df = pd.DataFrame({'Actual': test_y, 'Predicted': test_predictions})
print(result_df.head())

# Visualizing the actual vs predicted temperatures
plt.figure(figsize=(10, 5))
plt.plot(result_df['Actual'].values, label='Actual')
plt.plot(result_df['Predicted'].values, label='Predicted', linestyle='--')
plt.legend()
plt.title('Actual vs Predicted Temperature')
plt.show()

# Predicting future weather
def predict_future_weather(start_date, periods, frequency='D'):
    """
    Function to predict future weather data for a given number of periods (days, months, or years).
    """
    last_known_date = weather_df.index[-1]
    future_dates = pd.date_range(last_known_date, periods=periods, freq=frequency)
    
    # Use the most recent data to predict future weather conditions
    last_known_data = weather_df.tail(1).drop(['tempC'], axis=1)
    
    future_predictions = []
    for future_date in future_dates:
        prediction = regr.predict(last_known_data)[0]  # Predict temperature for the next period
        future_predictions.append(prediction)
        last_known_data.iloc[0, -1] = prediction  # Simulate future tempC in the last_known_data
    
    future_weather = pd.DataFrame({'Date': future_dates, 'Predicted_TempC': future_predictions})
    return future_weather

# Predicting the weather for the next 7 days (week)
future_weather_week = predict_future_weather(start_date=weather_df.index[-1], periods=7)
print(future_weather_week)

# Map visualization of Pune places

places = {
    'Kothrud': (18.5074, 73.8077),
    'Baner': (18.5582, 73.7860),
    'Hinjewadi': (18.5919, 73.7380),
    'Pimpri': (18.6283, 73.7997),
    'Wakad': (18.5983, 73.7622),
    'Aundh': (18.5645, 73.8077),
    'Magarpatta': (18.5199, 73.9260),
    'Hadapsar': (18.5018, 73.9260)
}

# Convert place data to GeoDataFrame for visualization
place_df = pd.DataFrame(places.items(), columns=['Place', 'Coordinates'])
place_df['Latitude'] = place_df['Coordinates'].apply(lambda x: x[0])
place_df['Longitude'] = place_df['Coordinates'].apply(lambda x: x[1])
place_df['geometry'] = place_df.apply(lambda row: Point(row['Longitude'], row['Latitude']), axis=1)

# Create a GeoDataFrame
gdf = gpd.GeoDataFrame(place_df, geometry='geometry')

# Load the base map of India from the downloaded shapefile
world = gpd.read_file("data/ne_110m_admin_0_countries.shp")

# Inspect the column names to find the one representing country names
print(world.columns)

# Filter for India based on the correct column (for example, 'ADMIN')
pune_map = world[world['ADMIN'] == 'India']  # Replace 'ADMIN' with actual column name if different
# Plotting map
fig, ax = plt.subplots(figsize=(10, 10))
pune_map.plot(ax=ax, color='lightgray')
gdf.plot(ax=ax, color='blue', marker='o', markersize=50, label='Pune Locations')

for x, y, label in zip(gdf['Longitude'], gdf['Latitude'], gdf['Place']):
    ax.text(x, y, label, fontsize=12, ha='right')

plt.title('Pune Weather Prediction: Location Overview')
plt.show()
