import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns



API_KEY = '123eecf8318e44efb40185031241211'
CITY = 'Khatmandu'

def get_real_time_weather(API_KEY, CITY):
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={CITY}"
    response = requests.get(url)
    data = response.json()
    
    # Print the data to check its structure
    # print(data)

    # Check and access the correct key
    humidity = data['current']['humidity']  # Assuming 'current' is the correct key
    temperature = data['current']['temp_c']  # Assuming 'temp_c' for temperature in Celsius
    rainfall = data['current']['precip_mm']  # Assuming 'precip_mm' for rainfall in mm
    
    return temperature, humidity, rainfall

weather_df = pd.read_csv('/Users/nidhi/Desktop/Crop_recommender/Data/Crop_recommendation.csv')
season_df = pd.read_excel('/Users/nidhi/Desktop/Crop_recommender/Data/Crop_Calendar_Data.xlsx', sheet_name=0)

weather_df.rename(columns={'label': 'Crop'}, inplace=True)
weather_df['Crop'] = weather_df['Crop'].str.strip().str.lower()
season_df['Crop'] = season_df['Crop'].str.strip().str.lower()

merged_df = pd.merge(weather_df, season_df, on='Crop', how='outer')
merged_df = merged_df.dropna(subset=['Crop'])

if 'month' not in merged_df.columns:
    merged_df['month'] = datetime.now().month  # Use current month as default

if 'next_month' not in merged_df.columns:
    merged_df['next_month'] = (merged_df['month'] % 12) + 1  # Calculate next month

# Define seasons based on months
def determine_season(month):
    if month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    elif month in [9, 10, 11]:
        return 'autumn'
    else:
        return 'winter'

merged_df['season'] = merged_df['month'].apply(determine_season)
merged_df['next_season'] = merged_df['next_month'].apply(determine_season)

# Prepare features and target
X = merged_df[['temperature', 'rainfall', 'humidity', 'month', 'next_month']].fillna(merged_df.select_dtypes(include=[float, int]).mean())
y = merged_df['Crop']

# Train-test split and model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Get real-time weather data
temperature, humidity, rainfall = get_real_time_weather(API_KEY, CITY)

if temperature is not None:
    # Prediction function using real-time data
    def recommend_crop(temperature, rainfall, humidity, month, season, next_season, day_weather=False):
        # Create a DataFrame with the same column names as the training data
        features = pd.DataFrame([[temperature, rainfall, humidity, month, (month % 12) + 1]],
                                columns=['temperature', 'rainfall', 'humidity', 'month', 'next_month'])

        if day_weather:
            # Predict for the current day
            current_day_prediction = model.predict(features)[0]
            return current_day_prediction
        else:
            # Predict for the current month
            current_month_prediction = model.predict(features)[0]

            # Predict for next month based on the next month's weather parameters
            next_month_features = pd.DataFrame([[temperature, rainfall, humidity, (month % 12) + 1, (month + 1) % 12 + 1]],
                                               columns=['temperature', 'rainfall', 'humidity', 'month', 'next_month'])
            next_month_prediction = model.predict(next_month_features)[0]

            # Predict for the current season
            current_season_crop = merged_df[merged_df['season'] == season]['Crop']
            current_season_prediction = current_season_crop.mode().iloc[0] if not current_season_crop.empty else "No crop recommendation for the current season"

            # Predict for the next season
            next_season_crop = merged_df[merged_df['season'] == next_season]['Crop']
            next_season_prediction = next_season_crop.mode().iloc[0] if not next_season_crop.empty else "No crop recommendation for the next season"

            return current_month_prediction, next_month_prediction, current_season_prediction, next_season_prediction

    # Get the current month and season
    current_month = datetime.now().month
    current_season = determine_season(current_month)
    next_season = determine_season((current_month % 12) + 3)

    # Get recommended crop based on real-time weather
    recommended_crop_day = recommend_crop(temperature, rainfall, humidity, current_month, current_season, next_season, day_weather=True)
    recommended_crop_current, recommended_crop_next, recommended_crop_current_season, recommended_crop_next_season = recommend_crop(temperature, rainfall, humidity, current_month, current_season, next_season)

    # Output results
    print(f"Recommended Crop for the current day: {recommended_crop_day}")
    print(f"Recommended Crop for the current month ({current_month}): {recommended_crop_current}")
    print(f"Recommended Crop for the next month ({(current_month % 12) + 1}): {recommended_crop_next}")
    print(f"Recommended Crop for the current season ({current_season}): {recommended_crop_current_season}")
    print(f"Recommended Crop for the next season ({next_season}): {recommended_crop_next_season}")
else:
    print("Failed to retrieve valid real-time weather data. Cannot make crop recommendation.")



# exploration
sns.set(style="whitegrid", palette="pastel")

def plot_feature_importance(model, feature_names):
    feature_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).query("Feature not in ['month', 'next_month']").sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(8, 4))
    sns.barplot(x='Importance', y='Feature', data=feature_df, palette="coolwarm")
    plt.title("Feature Importance (Excluding Month Features)")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.show()

plot_feature_importance(model, X.columns)



sns.set(style="whitegrid", palette="pastel")

# 1. Temperature vs. Crop Recommendation (Boxplot)
def plot_temperature_vs_crop(df):
    plt.figure(figsize=(15, 5))
    sns.boxplot(x='Crop', y='temperature', data=df)
    plt.xticks(rotation=90)
    plt.title("Temperature Range for Each Crop Recommendation")
    plt.xlabel("Crop")
    plt.ylabel("Temperature (°C)")
    plt.show()

# 2. Rainfall vs. Crop Recommendation (Boxplot)
def plot_rainfall_vs_crop(df):
    plt.figure(figsize=(15, 5))
    sns.boxplot(x='Crop', y='rainfall', data=df)
    plt.xticks(rotation=90)
    plt.title("Rainfall Levels for Each Crop Recommendation")
    plt.xlabel("Crop")
    plt.ylabel("Rainfall (mm)")
    plt.show()

# 3. Humidity vs. Crop Recommendation (Boxplot)
def plot_humidity_vs_crop(df):
    plt.figure(figsize=(15, 5))
    sns.boxplot(x='Crop', y='humidity', data=df)
    plt.xticks(rotation=90)
    plt.title("Humidity Range for Each Crop Recommendation")
    plt.xlabel("Crop")
    plt.ylabel("Humidity (%)")
    plt.show()

plot_temperature_vs_crop(merged_df)
plot_rainfall_vs_crop(merged_df)
plot_humidity_vs_crop(merged_df)
