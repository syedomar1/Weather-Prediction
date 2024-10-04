import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load dataset
df = pd.read_csv('pune.csv')

# Convert 'date_time' to datetime format
df['date_time'] = pd.to_datetime(df['date_time'], format='%d-%m-%Y %H:%M')
df['year'] = df['date_time'].dt.year
df['month'] = df['date_time'].dt.month
df['day'] = df['date_time'].dt.day
df['hour'] = df['date_time'].dt.hour

# Function to convert time to minutes
def convert_time_to_minutes(time_str):
    try:
        time_obj = datetime.strptime(time_str, '%I:%M %p')
        return time_obj.hour * 60 + time_obj.minute
    except:
        return np.nan

# Apply the function to convert moonrise, moonset, sunrise, and sunset columns
df['moonrise'] = df['moonrise'].apply(convert_time_to_minutes)
df['moonset'] = df['moonset'].apply(convert_time_to_minutes)
df['sunrise'] = df['sunrise'].apply(convert_time_to_minutes)
df['sunset'] = df['sunset'].apply(convert_time_to_minutes)

# One-Hot Encoding for the 'place' column
# encoder = OneHotEncoder(sparse_output=False)
# place_encoded = encoder.fit_transform(df[['place']])

# Create a DataFrame for the encoded 'place' data and concatenate with the original DataFrame
# place_encoded_df = pd.DataFrame(place_encoded, columns=encoder.get_feature_names_out(['place']), index=df.index)
# df = pd.concat([df, place_encoded_df], axis=1)

# Drop the original 'place' column (we have encoded it now)
df.drop(['place', 'date_time'], axis=1, inplace=True)

# Fill missing values (forward fill)
df.fillna(method='ffill', inplace=True)

# Variance and Mean of columns
print("Variance of the columns:")
print(df.var())
print("\nMean of the columns:")
print(df.mean())

# Correlation matrix visualization
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Correlation with temperature
corr_with_temp = df.corr()['tempC'].sort_values(ascending=False)
print("\nCorrelation with temperature:")
print(corr_with_temp)

# Features and target for modeling
X = df.drop(columns=['maxtempC', 'mintempC'])  # Features
y = df[['maxtempC', 'mintempC']]  # Targets

# Splitting data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training with RandomForestRegressor
temp_model = RandomForestRegressor(n_estimators=90, random_state=42)
temp_model.fit(X_train, y_train)

# Predict on the test set
y_pred = temp_model.predict(X_test)

# Evaluating model performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'R^2 Score: {r2:.2f}')

# Showing actual vs predicted temperatures
result_df = pd.DataFrame({'Actual Max Temp': y_test['maxtempC'], 
                          'Predicted Max Temp': y_pred[:, 0], 
                          'Actual Min Temp': y_test['mintempC'], 
                          'Predicted Min Temp': y_pred[:, 1]})
print(result_df.head())

# Visualizing the actual vs predicted temperatures
plt.figure(figsize=(10, 5))
plt.plot(result_df['Actual Max Temp'].values, label='Actual Max Temp')
plt.plot(result_df['Predicted Max Temp'].values, label='Predicted Max Temp', linestyle='--')
plt.plot(result_df['Actual Min Temp'].values, label='Actual Min Temp', linestyle='-.')
plt.plot(result_df['Predicted Min Temp'].values, label='Predicted Min Temp', linestyle=':')
plt.legend()
plt.title('Actual vs Predicted Max/Min Temperatures')
plt.show()

# Feature Importance Visualization
importance = temp_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.show()

# Prediction Function for a Given Date
def predict_temperature_for_date(date_time_str):
    try:
        # Strip any extra whitespace before parsing
        date_time_str = date_time_str.strip()
        future_date_time = datetime.strptime(date_time_str, '%d-%m-%Y %H:%M')
        year = future_date_time.year
        month = future_date_time.month
        day = future_date_time.day
        hour = future_date_time.hour

        total_snow_cm = max(0.1 - (year - 2020) * 0.005, 0)
        sun_hour = min(7.0 + (year - 2020) * 0.02, 10.0)
        uv_index = min(3 + (year - 2020) * 0.05, 11)
        humidity = max(60 - (year - 2020) * 0.2, 50)
        cloudcover = max(50 - (year - 2020) * 0.1, 40)
        precipMM = 2 + (year - 2020) * 0.1
        tempC = 10 + (year - 2020) * 0.05

        input_data = {
            'totalSnow_cm': [total_snow_cm],
            'sunHour': [sun_hour],
            'uvIndex': [uv_index],
            'uvIndex.1': [uv_index],
            'moon_illumination': [50],
            'moonrise': [12 * 60],
            'moonset': [12 * 60],
            'sunrise': [6 * 60],
            'sunset': [18 * 60],
            'DewPointC': [5],
            'FeelsLikeC': [10],
            'HeatIndexC': [15],
            'WindChillC': [5],
            'WindGustKmph': [15],
            'cloudcover': [cloudcover],
            'humidity': [humidity],
            'precipMM': [precipMM],
            'pressure': [1010],
            'tempC': [tempC],
            'visibility': [10],
            'winddirDegree': [180],
            'windspeedKmph': [10],
            'year': [year],
            'month': [month],
            'day': [day],
            'hour': [hour]
        }

        input_df = pd.DataFrame(input_data)
        temp_pred = temp_model.predict(input_df)
        print(f"Predicted Maximum Temperature: {temp_pred[0][0]+3:.2f}°C")
        print(f"Predicted Minimum Temperature: {temp_pred[0][1]+3:.2f}°C")

    except ValueError:
        print("Invalid date format. Please enter the date in 'dd-mm-yyyy HH:MM' format.")

# Predict Future Trends
def predict_future_trends(start_year=2020, end_year=2040):
    future_years = list(range(start_year, end_year + 1))
    max_temp_predictions = []
    min_temp_predictions = []
    # dummy_place_data = pd.DataFrame(0, index=range(1), columns=encoder.get_feature_names_out(['place']))
    for year in future_years:
        # Simulate trends for global warming effects
        total_snow_cm = max(0.1 - (year - 2020) * 0.005, 0)
        sun_hour = min(7.0 + (year - 2020) * 0.02, 10.0)
        uv_index = min(3 + (year - 2020) * 0.05, 11)
        humidity = max(60 - (year - 2020) * 0.2, 50)
        cloudcover = max(50 - (year - 2020) * 0.1, 40)
        tempC = 10 + (year - 2020) * 0.05

        dummy_data = {
            'totalSnow_cm': [total_snow_cm],
            'sunHour': [sun_hour],
            'uvIndex': [uv_index],
            'uvIndex.1': [uv_index],
            'moon_illumination': [50],
            'moonrise': [12 * 60],
            'moonset': [12 * 60],
            'sunrise': [6 * 60],
            'sunset': [18 * 60],
            'DewPointC': [5],
            'FeelsLikeC': [10],
            'HeatIndexC': [15],
            'WindChillC': [5],
            'WindGustKmph': [15],
            'cloudcover': [cloudcover],
            'humidity': [humidity],
            'precipMM': [2],
            'pressure': [1010],
            'tempC': [tempC],
            'visibility': [10],
            'winddirDegree': [180],
            'windspeedKmph': [10],
            'year': [year],
            'month': [6],  # Example month: June
            'day': [15],    # Example day: 15th
            'hour': [12]    # Example hour: noon
        }

        input_df = pd.DataFrame(dummy_data)
        # input_df = pd.concat([input_df, dummy_place_data], axis=1)
        temp_pred = temp_model.predict(input_df)
        max_temp_predictions.append(temp_pred[0][0])
        min_temp_predictions.append(temp_pred[0][1])
    future_df = pd.DataFrame({
        'Year': future_years,
        'Predicted Max Temp (°C)': max_temp_predictions,
        'Predicted Min Temp (°C)': min_temp_predictions
    })
    # Plot future predictions
    plt.figure(figsize=(12, 6))
    plt.plot(future_df['Year'], future_df['Predicted Max Temp (°C)'], label='Predicted Max Temp (°C)', color='red')
    plt.plot(future_df['Year'], future_df['Predicted Min Temp (°C)'], label='Predicted Min Temp (°C)', color='blue')
    plt.title('Predicted Temperature Trends (2020-2040)')
    plt.xlabel('Year')
    plt.ylabel('Temperature (°C)')
    plt.axhline(y=0, color='black', linewidth=0.8, linestyle='--')  # Horizontal line at y=0
    plt.legend()
    plt.grid()
    plt.show()
predict_future_trends(2020, 2040)
input_date = input("Enter a future date to predict temperature (dd-mm-yyyy HH:MM): ")
predict_temperature_for_date(input_date)

