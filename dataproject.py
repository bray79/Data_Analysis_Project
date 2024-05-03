#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 19:15:21 2024

@author: SwoopSolutions
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import calendar
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import time
import folium
from folium.plugins import HeatMap

# Set option to display all columns
pd.set_option('display.max_columns', None)

# Read in data from csv file and set values equal to \N ('\\N') to nan
df = pd.read_csv('chicago_2019_2022.csv', na_values=['\\N'])

# Print number of columns and rows of df DataFrame
print(df.shape)

# Get column names of df
column_names = df.columns.tolist()

# Get number of total columns
total_columns = len(column_names)

print('\ntotal columns: ',total_columns,'\n')
print('column names: ', column_names, '\n')

# Display number of unique countries, cities, and all unique states
print('unique country count: ',df['country'].nunique())
print('unique city count: ', df['city'].nunique())
print('unique states: ', df['state'].unique())

# Create a copy of DataFrame 'df' and assign it to 'df_1'
df_1 = df.copy()

print(df_1.info())

print('\ncount of missing values in each column before cleaning:')

# Get count of missing values for each column
missing_values_count = df_1.isnull().sum()

# Display count of missing values in each column
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(missing_values_count)

# Filter the copied DataFrame 'df_1' to include only the rows 
# where the value in the 'state' column is 'illinois'
df_1 = df_1[df_1['state'] == 'illinois']

print('\nunique states after filtering: ', df_1['state'].unique())
print('\nunique towns: ', df_1['town'].unique())

# Drop specified columns (30 cols) from DataFrame 'df_1'
df_1 = df_1.drop(columns=['country','city','state','id', 'days_windgust', 'days_winddir', 'days_precipcover',
                         'days_preciptype', 'days_feelslike', 'days_feelslikemin', 'days_feelslikemax',
                          'days_feelslike', 'days_temp', 'days_humidity', 'days_snow', 'days_dew', 
                          'days_precip', 'days_snowdepth', 'days_windspeed', 'days_visibility',
                          'days_conditions', 'days_cloudcover', 'days_precipprob', 'windgust',
                          'days_tempmax', 'days_tempmin', 'days_pressure', 'days_moonphase', 'feelslike',
                          'crash_hit_and_run', 'precip_type'
                         ])

# Reset column names to 'df_1' columns after dropping the specified columns
column_names = df_1.columns.tolist()

print('\ncolumn names after dropping unneeded columns: ', column_names)
print('\ntotal columns after dropping: ',len(column_names),'out of',total_columns,'\n')

# Create a copy of DataFrame 'df_1' and assign it to 'df_2'
df_2 = df_1.copy()

# Change column type to datetime with format mm/dd/yy
df_2['crash_date'] = pd.to_datetime(df_2['crash_date'], format='%m/%d/%y')

# Change columns to timedelta for analysis and comparison, removing '0 days' as it is not needed
df_2['crash_time'] = pd.to_timedelta(df_2['crash_time'] + ':00').astype(str).str.split().str[-1]
df_2['sunset'] = pd.to_timedelta(df_2['sunset']).astype(str).str.split().str[-1]
df_2['sunrise'] = pd.to_timedelta(df_2['sunrise']).astype(str).str.split().str[-1]

# Change object columns types to category for analysis
df_2['town'] = df_2['town'].astype('category')
df_2['crash_severity'] = df_2['crash_severity'].astype('category')
df_2['conditions'] = df_2['conditions'].astype('category')
df_2['most_severe_injury'] = df_2['most_severe_injury'].astype('category')
df_2['crash_type'] = df_2['crash_type'].astype('category')
df_2['contributory_cause'] = df_2['contributory_cause'].astype('category')
df_2['sec_contributory_cause'] = df_2['sec_contributory_cause'].astype('category')
df_2['traffic_control_device'] = df_2['traffic_control_device'].astype('category')
df_2['traffic_control_device_condition'] = df_2['traffic_control_device_condition'].astype('category')
df_2['road_defect'] = df_2['road_defect'].astype('category')

# Get all column names with type int64 or float64
numeric_column_names = df_2.select_dtypes(include=['int64', 'float64']).columns

# Get all column names with type category
categorical_column_names = df_2.select_dtypes(include='category').columns

# Get all column names with type datetime64[ns], timedelta64[ns], or object
datetime_column_names = df_2.select_dtypes(include=['datetime64[ns]', 'timedelta64[ns]', 'object']).columns

# Convert respective column names into DataFrames
df_numeric = df_2[numeric_column_names].copy()
df_categorical = df_2[categorical_column_names].copy()
df_datetime = df_2[datetime_column_names].copy()

# NUMERIC IMPUTATION WITH BAYESIAN RIDGE
imp_numeric = IterativeImputer(estimator=BayesianRidge(), max_iter=5, tol=1e-10, imputation_order='descending')
df_numeric_imputed = imp_numeric.fit_transform(df_numeric)

# Reassign numeric DataFrame to imputed numeric DataFrame
df_numeric_imputed = pd.DataFrame(df_numeric_imputed, columns=numeric_column_names)

# Convert column types to int32 for columns that should not have floating points
df_numeric_imputed['total_injured'] = df_numeric_imputed['total_injured'].astype('int32')
df_numeric_imputed['total_killed'] = df_numeric_imputed['total_killed'].astype('int32')
df_numeric_imputed['injury_incapacitated'] = df_numeric_imputed['injury_incapacitated'].astype('int32')
df_numeric_imputed['injury_non_incapacitated'] = df_numeric_imputed['injury_non_incapacitated'].astype('int32')
df_numeric_imputed['num_vehicles_in_crash'] = df_numeric_imputed['num_vehicles_in_crash'].astype('int32')
df_numeric_imputed['precipprob'] = df_numeric_imputed['precipprob'].astype('int32')
df_numeric_imputed['days_uvindex'] = df_numeric_imputed['days_uvindex'].astype('int32')

# CATEGORCIAL IMPUTATION WITH MODE
for col in df_categorical:
    df_categorical[col].fillna(df_categorical[col].mode()[0], inplace=True)

# Reset the indexes of numeric, categorical, and datetime DataFrames
df_numeric_imputed.reset_index(drop=True, inplace=True)
df_categorical.reset_index(drop=True, inplace=True)
df_datetime.reset_index(drop=True, inplace=True)

# Concat numeric, categorical, and datetime DataFrames together to form the imputed DataFrame 'df_imputed'
df_imputed = pd.concat([df_numeric_imputed, df_categorical, df_datetime], axis=1)

# Create new columns and subsets of df_imputed dataFrame
df_imputed['was_deadly'] = df_imputed['total_killed'] > 0
df_imputed['was_injury'] = df_imputed['total_injured'] > 0
df_imputed['was_dark'] = (df_imputed['crash_time'] < df_imputed['sunrise']) | (df_imputed['crash_time'] > df_imputed['sunset'])
df_days_visibility = df_imputed['visibility'].round()
rounded_temp = (df_imputed['temp'] / 10).round() * 10
rounded_temp_value_counts = rounded_temp.value_counts()
df_imputed['crash_year'] = df_imputed['crash_date'].dt.year
df_imputed['crash_month'] = df_imputed['crash_date'].dt.month_name()
df_imputed['crash_month'] = df_imputed['crash_month'].astype('category')
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

# Change column types to datetime with format HH:MM:SS for columns that contain time values
df_imputed['sunrise'] = pd.to_datetime(df_imputed['sunrise'], format='%H:%M:%S')
df_imputed['sunset'] = pd.to_datetime(df_imputed['sunrise'], format='%H:%M:%S')
df_imputed['crash_time'] = pd.to_datetime(df_imputed['crash_time'], format='%H:%M:%S')

# Get the crash time rounded to the hour for plotting and analysis
df_rounded_crash_time = df_imputed['crash_time'].dt.hour.round()
df_imputed['rounded_crash_time'] = df_imputed['crash_time'].dt.hour.round()

# Filter out records from the year 2018
df_filter_year = df_imputed[df_imputed['crash_year'] != 2018]

# Get count of crashes by each year and month
crash_count_yearly = df_filter_year.groupby(['crash_year', 'crash_month']).size().reset_index(name='crash_count')
crash_count_yearly['crash_month'] = pd.Categorical(crash_count_yearly['crash_month'], categories=month_order, ordered=True)
pivoted_crash_count_yearly = crash_count_yearly.pivot(index='crash_year', columns='crash_month', values='crash_count').fillna(0)

# Get crash types that resulted in injury
injury_crashes_types = df_imputed.loc[df_imputed['was_injury'] == True, 'crash_type']
# Get crash types that resuted in death
deadly_crashes_types = df_imputed.loc[df_imputed['was_deadly'] == True, 'crash_type']

# Get weather conditions that resulted in injury
injury_days_conditions_types = df_imputed.loc[df_imputed['was_injury'] == True, 'conditions']
# Get weather conditions that resulted in death
deadly_days_conditions_types = df_imputed.loc[df_imputed['was_deadly'] == True, 'conditions']

# Get traffic control devices that resulted in injury
injury_traffic_control_devices = df_imputed.loc[df_imputed['was_injury'] == True, 'traffic_control_device']

# Get records where total_killed and total_injured are greater than 0
fd = df_imputed[(df_imputed['total_killed'] >= 0) & (df_imputed['total_injured'] >= 0)]
# Get total killed and injured by the crash time hour
hourly_data = fd.groupby(fd['crash_time'].dt.hour)[['total_killed', 'total_injured']].sum()
# Get total killed and injured by crash type
by_crash_type_data = fd.groupby(fd['crash_type'])[['total_killed', 'total_injured']].sum()

# Assigning the day of the week to a new column 'crash_day' based on the 'crash_date' column
df_imputed['crash_day'] = df_imputed['crash_date'].dt.day_name()
# Converting the 'crash_day' column to a categorical type for better visualization
df_imputed['crash_day'] = df_imputed['crash_day'].astype('category')
# Get the count of the occurrences of each day of the week
crash_day_counts = df_imputed['crash_day'].value_counts()
# Create a list of days of the week in order
sorted_days = list(calendar.day_name)

# Grouping and summing the occurrences of fatal and incapacitating injuries by day of the week
fatal_incapacitating_counts = df_imputed['most_severe_injury'].isin(['fatal', 'incapacitating injury']).groupby(df_imputed['crash_day']).sum()

# Creating a DataFrame to store the weekday and corresponding total fatal/incapacitating injuries
fatal_incapacitating_data = pd.DataFrame({'Weekday': fatal_incapacitating_counts.index,
                                          'Total Fatal/Incapacitating': fatal_incapacitating_counts.values})

# Converting the 'Weekday' column to a categorical type with ordered days
fatal_incapacitating_data['Weekday'] = pd.Categorical(fatal_incapacitating_data['Weekday'], categories=sorted_days, ordered=True)

# Sorting the DataFrame by weekday for better visualization
fatal_incapacitating_data = fatal_incapacitating_data.sort_values('Weekday')

# Filtering out rows where contributory cause is 'unable to determine' or 'not applicable'
filtered_contributory_causes = df_imputed[~df_imputed['contributory_cause'].isin(['unable to determine', 'not applicable'])]

# Selecting records where there was an injury
injury_df = df_imputed[df_imputed['was_injury'] == True]

# Get the top 15 contributory causes of accidents
top_contributory_causes = filtered_contributory_causes['contributory_cause'].value_counts().head(15)
# Get the top 15 crash types
top_crash_types = df_imputed['crash_type'].value_counts().head(15)
# Get the top 15 towns where accidents occurred
top_towns = df_imputed['town'].value_counts().head(15)
# Get the top 15 traffic control devices at accident locations
top_traffic_control_devices = df_imputed['traffic_control_device'].value_counts().head(15)
# Get the top 10 traffic control devices at accidents with injuries
top_traffic_control_devices_injury = injury_df['traffic_control_device'].value_counts().head(10)

# Filtering rows where the contributory cause is 'weather'
weather_data = df_imputed[df_imputed['contributory_cause'] == 'weather']

# Get the count of the occurrences of different weather conditions
weather_conditions = weather_data['conditions'].value_counts()

# Selecting columns with categorical data
categorical_columns = df_imputed.select_dtypes(include='category')
# Selecting columns with numerical data
numerical_columns = df_imputed.select_dtypes(include=['float64', 'int32', 'bool'])

# Calculating the correlation matrix for numerical columns
correlation_matrix = numerical_columns.corr()

print('\ncount of missing values in each column after cleaning:')

# Get count of missing values for each column
missing_values_count = df_imputed.isnull().sum()

# Display count of missing values in each column
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(missing_values_count)

# Display information of cleaned Dataframe 'df_imputed'
print('\nCleaned data types:\n', df_imputed.dtypes)
print('\nSample of cleaned data:\n', df_imputed.head(5))
print('\nShape of DataFrame df_imputed: ', df_imputed.shape, '\n')
print(df_imputed.info())
print(df_imputed.describe())

tester_categorical_columns = categorical_columns.drop(columns=['most_severe_injury', 'crash_month', 'crash_day', 'town'])
tester_categorical_columns = pd.get_dummies(tester_categorical_columns)

df_feat = pd.concat([numerical_columns, tester_categorical_columns], axis=1)

df_tester = df_feat.drop(columns=['total_killed', 'total_injured', 'injury_incapacitated', 'injury_non_incapacitated', 'was_deadly', 'crash_year'])

# Get 54,568 records where was_injury is True
was_injury_true_df = df_tester[df_tester['was_injury'] == True]
was_injury_true_sample = was_injury_true_df.sample(n=54568, random_state=42)

# Get 54,568 records where was_injury is False
was_injury_false_df = df_tester[df_tester['was_injury'] == False]
was_injury_false_sample = was_injury_false_df.sample(n=54568, random_state=42)

# Combine both samples to form the DataFrame 'df_model' that will be used for predictive analysis
df_model = pd.concat([was_injury_true_sample, was_injury_false_sample])
# Reset the index of 'df_model' DataFrame
df_model.reset_index(drop=True, inplace=True)

#   K-FOLD CROSS VALIDATION
# Establish X and y variables
X = df_model.drop('was_injury', axis=1)  # Features
y = df_model['was_injury']  # Target variable

# Initialize KFold with number of folds - 5
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=0)

# Initialize KNN classifier
classifier = KNeighborsClassifier(n_neighbors = 50, metric = 'manhattan', p = 2, weights='distance', algorithm='auto')

# Initialize StandardScaler
scaler = StandardScaler()

# Lists to store evaluation results
accuracy_scores = []

# Perform k-fold cross-validation
for train_index, test_index in kf.split(X):
    # Split data into training and test sets
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Standardize features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit classifier on standardized training data
    classifier.fit(X_train_scaled, y_train)
    
    # Predict on standardized test data
    y_pred = classifier.predict(X_test_scaled)
    
    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

# Calculate mean and standard deviation of accuracy scores
mean_accuracy = np.mean(accuracy_scores)
std_accuracy = np.std(accuracy_scores)

# Print results from K-Fold Cross Validation
print('\nK-Fold Cross Validation Results:')
print('Mean Accuracy:', mean_accuracy)
print('Standard Deviation of Accuracy:', std_accuracy)
print('Accuracy Scores: ', accuracy_scores)

# independent variables
X = df_model.drop('was_injury', axis=1)
# dependent variable
y = df_model['was_injury']

# Split data into training and test sets with a training size of 80% and testing size of 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize values for classifier
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initialize KNN classifier
classifier = KNeighborsClassifier(n_neighbors = 50, metric = 'manhattan', p = 2, weights='distance', algorithm='auto')

# Start timer
t0 = time.time()
# Fit classifier on standardized training data
classifier.fit(X_train, y_train)
# Print time elapsed after training the model
print('Training Time:', time.time()-t0)

# Print the parameters passed to the KNN classifier
print(classifier.get_params())

# Start timer
t0 = time.time()
# Predict on standardized test data
y_pred = classifier.predict(X_test)
# Print time elapsed after training the model
print('Prediction Time:', time.time()-t0)

# Generate the confusion matrix using the predicted and actual labels
cm = confusion_matrix(y_test, y_pred, labels=classifier.classes_)

# Print the confusion matrix
print('\nConfusion Matrix:\n', cm)
print('\n')

# Print the classification report showing precision, recall, F1-score, and support for each class
print(classification_report(y_test, y_pred))

######################   FIGURES / PLOTS  #############################

#               CRASH LOCATIONS HEAT MAP

# Create a map centered around the mean latitude and longitude of the data
map_center = [df_imputed['lattitude'].mean(), df_imputed['longitude'].mean()]
m = folium.Map(location=map_center, zoom_start=12)

# Create a HeatMap layer
heat_data = [[row['lattitude'], row['longitude']] for index, row in df_imputed.iterrows()]
HeatMap(heat_data).add_to(m)

# Save the map
m.save('crash_map.html')


#               PAIR PLOT

# Sampling 500 rows of numeric data where 'was_injury' is True and False
sample_was_injury_numeric_data = numerical_columns.loc[numerical_columns['was_injury'] == True].sample(500).reset_index(drop=True)
sample_not_injury_numeric_data = numerical_columns.loc[numerical_columns['was_injury'] == False].sample(500).reset_index(drop=True)

# Removing duplicate columns from each sample
sample_was_injury_numeric_data = sample_was_injury_numeric_data.loc[:, ~sample_was_injury_numeric_data.columns.duplicated()]
sample_not_injury_numeric_data = sample_not_injury_numeric_data.loc[:, ~sample_not_injury_numeric_data.columns.duplicated()]

# Concatenating the two samples vertically
sample_numeric_data = pd.concat([sample_was_injury_numeric_data, sample_not_injury_numeric_data], ignore_index=True)

# Selecting 5 random attributes and 'was_injury', then creating a DataFrame with these columns
sample_selected_attributes = ['was_injury'] + sample_numeric_data.sample(5, axis=1, random_state=302).columns.tolist()
sample_selected_data = sample_numeric_data[sample_selected_attributes]

# Creating pair plots to visualize relationships between selected attributes, colored by 'was_injury'
plt.figure(figsize=(12, 5))
sns.pairplot(sample_selected_data, hue='was_injury')
plt.show()


#               CONFUSION MATRIX

# Displaying the confusion matrix using ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
disp.plot()
disp.ax_.set(xlabel='Predicted was_injury', ylabel='Actual was_injury')
plt.title('Actual vs Predicted Confusion Matrix')
plt.show()


# Set 'crash_date' as the index
df_imputed.set_index('crash_date', inplace=True)
# Get the number of crashes by month
monthly_crashes = df_imputed.resample('M').size()

# Plotting the number of crashes per month
monthly_crashes.plot(figsize=(15, 5), title='Number of Crashes per Month')
plt.xlabel('Crash Date')
plt.ylabel('Count')
plt.grid(axis='both')
plt.show()

# Plotting the distribution of crashes during different times of the day
plt.figure(figsize=(10, 6))
plt.hist(df_imputed['crash_time'].dt.hour, bins=24, edgecolor='black', alpha=0.7)
plt.title('Distribution of Crashes During Different Times of the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Crashes')
plt.xticks(range(24), [hour for hour in range(24)], rotation=0)
plt.grid(axis='y')
plt.show()

# Plotting the total killed and total injured by hour of the day
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
hourly_data['total_killed'].plot(kind='bar', color='lightsalmon', edgecolor='black')
plt.title('Total Killed by Hour')
plt.xlabel('Hour of the Day')
plt.xticks(rotation=0)
plt.ylabel('Count')
plt.grid(axis='y')

plt.subplot(1, 2, 2)
hourly_data['total_injured'].plot(kind='bar', color='lightsteelblue', edgecolor='black')
plt.title('Total Injured by Hour')
plt.xlabel('Hour of the Day')
plt.xticks(rotation=0)
plt.ylabel('Count')
plt.grid(axis='y')

plt.tight_layout()
plt.show()

# Plotting the number of crashes by day of the week and fatal/incapacitating injuries by weekday
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
crash_day_counts.loc[sorted_days].plot(kind='bar', color='lightsteelblue', edgecolor='black')
plt.title('Number of Crashes by Day of the Week')
plt.xlabel('Day of the Week')
plt.xticks(rotation=0)
plt.ylabel('Number of Crashes')
plt.grid(axis='y')

plt.subplot(1, 2, 2)
plt.bar(fatal_incapacitating_data['Weekday'], fatal_incapacitating_data['Total Fatal/Incapacitating'], color='lightsalmon', edgecolor='black')
plt.title('Fatal and Incapacitating Injuries by Weekday')
plt.xlabel('Day of the Week')
plt.ylabel('Total Fatal/Incapacitating Injuries')
plt.grid(axis='y')

plt.tight_layout()
plt.show()

# Plotting the top 15 contributory causes of crashes
plt.figure(figsize=(12, 6))
top_contributory_causes.plot(kind='bar', color='lightsteelblue', edgecolor='black', alpha=0.7)
plt.title('Top 15 Contributory Causes of Crashes')
plt.xlabel('Contributory Cause')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.show()

# Plotting the distribution of crashes by weather condition in crashes caused by weather
plt.figure(figsize=(10, 6))
weather_conditions.plot(kind='bar', color='lightsteelblue', edgecolor='black')
plt.title('Crash Distribution by Weather Condition in Crashes Caused by Weather')
plt.xlabel('Weather Conditions')
plt.ylabel('Number of Crashes')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Plotting the top 15 crash types by frequency
plt.figure(figsize=(12, 6))
top_crash_types.plot(kind='bar', color='lightsteelblue', edgecolor='black', alpha=0.7)
plt.title('Top 15 Crash Types by Frequency')
plt.xlabel('Crash Type')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.show()

# Plotting the total killed and total injured by crash type
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
by_crash_type_data['total_killed'].plot(kind='bar', color='lightsalmon', edgecolor='black')
plt.title('Total Killed by Crash Type')
plt.xlabel('Crash Type')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Count')
plt.grid(axis='y')

plt.subplot(1, 2, 2)
by_crash_type_data['total_injured'].plot(kind='bar', color='lightsteelblue', edgecolor='black')
plt.title('Total Injured by Crash Type')
plt.xlabel('Crash Type')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Count')
plt.grid(axis='y')

plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 8))
# Creating a count plot to visualize the distribution of crashes by weather condition and darkness
sns.countplot(x='conditions', hue='was_dark', data=df_imputed,\
              order=df_imputed['conditions'].value_counts().index, palette=['lightsalmon', 'lightsteelblue'],\
                  edgecolor='black')
plt.title('Crash Distribution by Weather Condition and Darkness')
plt.xlabel('Conditions')
plt.ylabel('Number of Crashes')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Was Dark', loc='upper right', fontsize='large')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(12, 6))
# Plotting the top 15 towns with the highest crash frequency
top_towns.plot(kind='bar', color='lightsteelblue', edgecolor='black')
plt.title('Top 15 Towns with Highest Crash Frequency')
plt.xlabel('Town')
plt.ylabel('Number of Crashes')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 8))
# Creating a heatmap to visualize the correlation matrix of numerical data
sns.heatmap(correlation_matrix, annot=True, annot_kws={'size': 8}, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Numerical Data')
plt.show()

# Set the figure size
plt.figure(figsize=(12, 5))
# Creating a bar plot to display the top 15 traffic control devices and their crash counts
top_traffic_control_devices.plot(kind='bar', color='lightsteelblue', edgecolor='black')
# Adding labels and title
plt.xlabel('Traffic Control Devices', fontsize=12)
plt.ylabel('Count of Crashes', fontsize=12)
plt.title('Top 15 Traffic Control Devices and Count of Crashes', fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis='y')

plt.figure(figsize=(12, 5))
# Creating a bar plot to display the top 10 traffic control devices and their crash counts with injuries
top_traffic_control_devices_injury.plot(kind='bar', color='lightsalmon', edgecolor='black')
# Adding labels and title
plt.xlabel('Traffic Control Devices', fontsize=12)
plt.ylabel('Count of Crashes with Injuries', fontsize=12)
plt.title('Top 10 Traffic Control Devices and Count of Crashes with Injuries', fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis='y')

#plt.tight_layout()
plt.show()

# Set the figure size
plt.figure(figsize=(10, 6))
# Creating a grouped bar plot to visualize the relationship between temperature and count of crashes
rounded_temp_value_counts.plot(kind='bar', color='lightsteelblue', edgecolor='black')
# Adding labels and title
plt.xlabel('Rounded Temperature in Farenheit', fontsize=14)
plt.ylabel('Count of Crashes', fontsize=14)
plt.title('Temperature and Count of Crashes', fontsize=16)
# Rotating x-axis labels for better readability
plt.xticks(fontsize=12, rotation=0)
plt.yticks(fontsize=12)
plt.grid(axis='y')

# Adjusting layout to prevent overlapping labels
plt.tight_layout()
# Showing plot
plt.show()
