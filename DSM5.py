import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Path to the CSV file
file_path = 'US_Accidents_March23.csv'

# Load the dataset
try:
    df = pd.read_csv(file_path, on_bad_lines='skip')
except pd.errors.ParserError as e:
    print(f"ParserError: {e}")
    exit()
except FileNotFoundError as e:
    print(f"FileNotFoundError: {e}")
    exit()

# Check for missing values
print(df.isnull().sum())

# Drop rows with significant missing data or fill with appropriate values
df.dropna(subset=['Start_Time', 'End_Time', 'Weather_Condition'], inplace=True)

# Convert date columns to datetime with error handling
df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')

# Drop rows where datetime conversion failed
df.dropna(subset=['Start_Time', 'End_Time'], inplace=True)

# Distribution of accidents over time
df['Hour'] = df['Start_Time'].dt.hour
sns.histplot(df['Hour'], bins=24, kde=True)
plt.title('Distribution of Accidents by Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Accidents')
plt.show()

# Distribution by weather conditions
sns.countplot(x='Weather_Condition', data=df)
plt.title('Distribution of Accidents by Weather Condition')
plt.xticks(rotation=90)
plt.xlabel('Weather Condition')
plt.ylabel('Number of Accidents')
plt.show()

# Visualizing Accident Hotspots
# Sample subset for performance (optional)
df_sample = df.sample(n=10000, random_state=1)

# Create a base map
base_map = folium.Map(location=[df_sample['Start_Lat'].mean(), df_sample['Start_Lng'].mean()], zoom_start=5)

# Add heatmap
heat_data = [[row['Start_Lat'], row['Start_Lng']] for index, row in df_sample.iterrows()]
HeatMap(heat_data).add_to(base_map)

# Save map
base_map.save("accident_hotspots.html")

# Correlation analysis
corr = df[['Hour', 'Severity', 'Weather_Condition']].apply(lambda x: pd.factorize(x)[0]).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation between Factors')
plt.show()

# Further analysis and machine learning (optional)

# Feature engineering
df['Day_of_Week'] = df['Start_Time'].dt.dayofweek

# Select features and target variable
features = df[['Hour', 'Day_of_Week', 'Weather_Condition']].apply(lambda x: pd.factorize(x)[0])
target = df['Severity']

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))