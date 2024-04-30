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
from sklearn.linear_model import BayesianRidge
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Set option to display all columns
pd.set_option('display.max_columns', None)

# Read in data from csv file and set values equal to \N ('\\N') to nan
df = pd.read_csv('chicago_2019_2022.csv', na_values=['\\N'])

print(df.shape)

column_names = df.columns.tolist()
total_columns = len(column_names)

print('\ntotal columns: ',total_columns,'\n')
print('column names: ', column_names, '\n')

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

# Filter the copied DataFrame 'df_1' to include only rows where the value in the 'state' column is 'illinois'
df_1 = df_1[df_1['state'] == 'illinois']

print('\nunique states after filtering: ', df_1['state'].unique())
print('\nunique towns: ', df_1['town'].unique())

# Drop specified columns (27 cols) from DataFrame 'df_1'
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

# Print the name and data type of each attribute/column
#print('uncleaned data types:\n', df_1.dtypes)

df_2 = df_1.copy()

df_2['crash_date'] = pd.to_datetime(df_2['crash_date'], format='%m/%d/%y')
# Change columns to timedelta for analysis and comparison, removing '0 days' as it is not needed
df_2['crash_time'] = pd.to_timedelta(df_2['crash_time'] + ':00').astype(str).str.split().str[-1]
df_2['sunset'] = pd.to_timedelta(df_2['sunset']).astype(str).str.split().str[-1]
df_2['sunrise'] = pd.to_timedelta(df_2['sunrise']).astype(str).str.split().str[-1]

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

numeric_column_names = df_2.select_dtypes(include=['int64', 'float64']).columns
categorical_column_names = df_2.select_dtypes(include='category').columns
datetime_column_names = df_2.select_dtypes(include=['datetime64[ns]', 'timedelta64[ns]', 'object']).columns

df_numeric = df_2[numeric_column_names].copy()
df_categorical = df_2[categorical_column_names].copy()
df_datetime = df_2[datetime_column_names].copy()

# NUMERIC IMPUTATION
imp_numeric = IterativeImputer(estimator=BayesianRidge(), max_iter=5, tol=1e-10, imputation_order='descending')
df_numeric_imputed = imp_numeric.fit_transform(df_numeric)
df_numeric_imputed = pd.DataFrame(df_numeric_imputed, columns=numeric_column_names)

df_numeric_imputed['total_injured'] = df_numeric_imputed['total_injured'].astype('int32')
df_numeric_imputed['total_killed'] = df_numeric_imputed['total_killed'].astype('int32')
df_numeric_imputed['injury_incapacitated'] = df_numeric_imputed['injury_incapacitated'].astype('int32')
df_numeric_imputed['injury_non_incapacitated'] = df_numeric_imputed['injury_non_incapacitated'].astype('int32')
df_numeric_imputed['num_vehicles_in_crash'] = df_numeric_imputed['num_vehicles_in_crash'].astype('int32')
df_numeric_imputed['precipprob'] = df_numeric_imputed['precipprob'].astype('int32')
df_numeric_imputed['days_uvindex'] = df_numeric_imputed['days_uvindex'].astype('int32')

for col in df_categorical:
    df_categorical[col].fillna(df_categorical[col].mode()[0], inplace=True)

df_numeric_imputed.reset_index(drop=True, inplace=True)
df_categorical.reset_index(drop=True, inplace=True)
df_datetime.reset_index(drop=True, inplace=True)

df_imputed = pd.concat([df_numeric_imputed, df_categorical, df_datetime], axis=1)

# Create new columns and subsets of df_imputed dataFrame
df_imputed['was_deadly'] = df_imputed['total_killed'] > 0
df_imputed['was_injury'] = df_imputed['total_injured'] > 0
df_imputed['was_dark'] = (df_imputed['crash_time'] < df_imputed['sunrise']) | (df_imputed['crash_time'] > df_imputed['sunset'])
df_days_visibility = df_imputed['visibility'].round()
df_days_temp = (df_imputed['temp'] / 10).round() * 10
df_crash_type_injury = df_imputed.groupby(['crash_type', 'was_injury']).size().reset_index(name='count')
df_weather_injury = df_imputed.groupby(['conditions', 'was_injury']).size().reset_index(name='count')
df_control_condition_injury = df_imputed.groupby(['traffic_control_device', 'was_injury']).size().reset_index(name='count')

df_imputed['sunrise'] = pd.to_datetime(df_imputed['sunrise'], format='%H:%M:%S')
df_imputed['sunset'] = pd.to_datetime(df_imputed['sunrise'], format='%H:%M:%S')
df_imputed['crash_time'] = pd.to_datetime(df_imputed['crash_time'], format='%H:%M:%S')
df_rounded_crash_time = df_imputed['crash_time'].dt.hour.round()
df_imputed['rounded_crash_time'] = df_imputed['crash_time'].dt.hour.round()

df_imputed['crash_year'] = df_imputed['crash_date'].dt.year
df_imputed['crash_month'] = df_imputed['crash_date'].dt.month_name()
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

df_filter_year = df_imputed[df_imputed['crash_year'] != 2018]
crash_count_yearly = df_filter_year.groupby(['crash_year', 'crash_month']).size().reset_index(name='crash_count')
crash_count_yearly['crash_month'] = pd.Categorical(crash_count_yearly['crash_month'], categories=month_order, ordered=True)
pivoted_crash_count_yearly = crash_count_yearly.pivot(index='crash_year', columns='crash_month', values='crash_count').fillna(0)

injury_crashes_types = df_imputed.loc[df_imputed['was_injury'] == True, 'crash_type']
deadly_crashes_types = df_imputed.loc[df_imputed['was_deadly'] == True, 'crash_type']

injury_days_conditions_types = df_imputed.loc[df_imputed['was_injury'] == True, 'conditions']
deadly_days_conditions_types = df_imputed.loc[df_imputed['was_deadly'] == True, 'conditions']

injury_traffic_control_devices = df_imputed.loc[df_imputed['was_injury'] == True, 'traffic_control_device']

fd = df_imputed[(df_imputed['total_killed'] >= 0) & (df_imputed['total_injured'] >= 0)]
hourly_data = fd.groupby(fd['crash_time'].dt.hour)[['total_killed', 'total_injured']].sum()
by_crash_type_data = fd.groupby(fd['crash_type'])[['total_killed', 'total_injured']].sum()

df_imputed['crash_day'] = df_imputed['crash_date'].dt.day_name()
crash_day_counts = df_imputed['crash_day'].value_counts()
sorted_days = list(calendar.day_name)

fatal_incapacitating_counts = df_imputed['most_severe_injury'].isin(['fatal',\
                                         'incapacitating injury']).groupby(df_imputed['crash_day']).sum()

fatal_incapacitating_data = pd.DataFrame({'Weekday': fatal_incapacitating_counts.index,\
                                          'Total Fatal/Incapacitating': fatal_incapacitating_counts.values})
                                           
fatal_incapacitating_data['Weekday'] = pd.Categorical(fatal_incapacitating_data['Weekday'],\
                                                      categories=sorted_days, ordered=True)
    
fatal_incapacitating_data = fatal_incapacitating_data.sort_values('Weekday')

filtered_contributory_causes = df_imputed[~df_imputed['contributory_cause'].isin(['unable to determine',\
                                                                                'not applicable'])]
    
top_contributory_causes = filtered_contributory_causes['contributory_cause'].value_counts().head(15)
top_crash_types = df_imputed['crash_type'].value_counts().head(15)
top_towns = df_imputed['town'].value_counts().head(15)

weather_data = df_imputed[df_imputed['contributory_cause'] == 'weather']
weather_conditions = weather_data['conditions'].value_counts()

categorical_columns = df_imputed.select_dtypes(include='category')
numerical_columns = df_imputed.select_dtypes(include=['float64', 'int32', 'bool'])
correlation_matrix = numerical_columns.corr()

print('\ncount of missing values in each column after cleaning:')

# Get count of missing values for each column
missing_values_count = df_imputed.isnull().sum()

# Display count of missing values in each column
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(missing_values_count)

print('\nCleaned data types:\n', df_imputed.dtypes)
print('\nSample of cleaned data:\n', df_imputed.head(5))
print('\nShape of DataFrame df_imputed: ', df_imputed.shape, '\n')
print(df_imputed.info())
print(df_imputed.describe())

"""
sample_was_deadly_numeric_data = numerical_columns.loc[numerical_columns['was_deadly'] == True].sample(50).reset_index(drop=True)
sample_not_deadly_numeric_data = numerical_columns.loc[numerical_columns['was_deadly'] == False].sample(50).reset_index(drop=True)

# Remove duplicate columns from each sample
sample_was_deadly_numeric_data = sample_was_deadly_numeric_data.loc[:,~sample_was_deadly_numeric_data.columns.duplicated()]
sample_not_deadly_numeric_data = sample_not_deadly_numeric_data.loc[:,~sample_not_deadly_numeric_data.columns.duplicated()]

# Concatenate the two samples vertically
sample_numeric_data = pd.concat([sample_was_deadly_numeric_data, sample_not_deadly_numeric_data], ignore_index=True)

plt.figure(figsize=(12, 5))
sns.pairplot(sample_numeric_data, hue='was_deadly')
plt.show()
"""

tester_categorical_columns = categorical_columns.drop(columns=['most_severe_injury', 'sec_contributory_cause'])
tester_categorical_columns = pd.get_dummies(tester_categorical_columns)
df_feat = pd.concat([numerical_columns, tester_categorical_columns], axis=1)

df_tester = df_feat.drop(columns=['total_killed', 'total_injured', 'injury_incapacitated', 'injury_non_incapacitated', 'was_deadly'])

# independent variables
X = df_tester.drop('was_injury', axis=1)
# dependent variable
y = df_tester['was_injury']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Standardiz values for KNN classifier
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#The default metric is minkowski, and with p=2 is equivalent to the standard Euclidean metric.
classifier = KNeighborsClassifier(n_neighbors = 10, metric = 'minkowski', p = 2, weights='distance', algorithm='auto')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print(cm)
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Precision: ', precision_score(y_test, y_pred))
print('Recall: ', recall_score(y_test, y_pred))
print(classification_report(y_test,y_pred))

######################   FIGURES / PLOTS  #############################

fig_labels = ['No', 'Yes']

#                           NEW FIGURES
# Plot for total killed
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
hourly_data['total_killed'].plot(kind='bar', color='lightsalmon', edgecolor='black')
plt.title('Total Killed by Hour')
plt.xlabel('Hour of the Day')
plt.xticks(rotation=0)
plt.ylabel('Count')
plt.grid(axis='x')

# Plot for total injured
plt.subplot(1, 2, 2)
hourly_data['total_injured'].plot(kind='bar', color='lightsteelblue', edgecolor='black')
plt.title('Total Injured by Hour')
plt.xlabel('Hour of the Day')
plt.xticks(rotation=0)
plt.ylabel('Count')
plt.grid(axis='x')

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(df_imputed['crash_time'].dt.hour, bins=24, edgecolor='black', alpha=0.7)
plt.title('Distribution of Crashes During Different Times of the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Crashes')
plt.xticks(range(24), [hour for hour in range(24)], rotation=0)
plt.grid(axis='x')
plt.show()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
crash_day_counts.loc[sorted_days].plot(kind='bar', color='lightsteelblue', edgecolor='black')
plt.title('Number of Crashes by Day of the Week')
plt.xlabel('Day of the Week')
plt.xticks(rotation=0)
plt.ylabel('Number of Crashes')
plt.grid(axis='x')

plt.subplot(1, 2, 2)
plt.bar(fatal_incapacitating_data['Weekday'], fatal_incapacitating_data['Total Fatal/Incapacitating'], color='lightsalmon', edgecolor='black')
plt.title('Fatal and Incapacitating Injuries by Weekday')
plt.xlabel('Day of the Week')
plt.ylabel('Total Fatal/Incapacitating Injuries')
plt.grid(axis='x')

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
top_contributory_causes.plot(kind='bar', color='lightsteelblue', edgecolor='black', alpha=0.7)
plt.title('Top 15 Contributory Causes of Crashes')
plt.xlabel('Contributory Cause')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='x')
plt.show()

plt.figure(figsize=(10, 6))
weather_conditions.plot(kind='bar', color='lightsteelblue', edgecolor='black')
plt.title('Crash Distribution by Weather Condition in Crashes Caused by Weather')
plt.xlabel('Weather Conditions')
plt.ylabel('Number of Crashes')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='x')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
top_crash_types.plot(kind='bar', color='lightsteelblue', edgecolor='black', alpha=0.7)
plt.title('Top 15 Crash Types by Frequency')
plt.xlabel('Crash Type')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='x')
plt.show()

# Plot for total killed
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
by_crash_type_data['total_killed'].plot(kind='bar', color='lightsalmon', edgecolor='black')
plt.title('Total Killed by Crash Type')
plt.xlabel('Crash Type')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Count')
plt.grid(axis='x')

# Plot for total injured
plt.subplot(1, 2, 2)
by_crash_type_data['total_injured'].plot(kind='bar', color='lightsteelblue', edgecolor='black')
plt.title('Total Injured by Crash Type')
plt.xlabel('Crash Type')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Count')
plt.grid(axis='x')

plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 8))
sns.countplot(x='conditions', hue='crash_severity', data=df_imputed,\
              order=df_imputed['conditions'].value_counts().index, palette=['lightsalmon', 'lightsteelblue'],\
                  edgecolor='black')
plt.title('Crash Distribution by Weather Condition and Crash Severity')
plt.xlabel('Conditions')
plt.ylabel('Number of Crashes')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Crash Severity', loc='upper right')
plt.show()

plt.figure(figsize=(12, 6))
top_towns.plot(kind='bar', color='lightsteelblue', edgecolor='black')
plt.title('Top 15 Towns with Highest Crash Frequency')
plt.xlabel('Town')
plt.ylabel('Number of Crashes')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='x')
plt.show()

pivoted_crash_count_yearly.plot(figsize=(12, 6), marker='o')
plt.title('Number of Crashes by Month and Year')
plt.xlabel('Year')
plt.ylabel('Number of Crashes')
plt.xticks(pivoted_crash_count_yearly.index)
plt.legend(title='Month', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, annot_kws={'size': 8}, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Numerical Data')
plt.show()


#                           FIGURE 3
# Set the figure size
plt.figure(figsize=(10, 6))
# Create horizontal bar plot
sns.countplot(y='traffic_control_device', data=df_imputed, color='blue')
# Add labels and title
plt.xlabel('Count of Crashes', fontsize=12)
plt.ylabel('Traffic Control Devices', fontsize=12)
plt.title('Traffic Control Devices and Count of Crashes', fontsize=14)
plt.xticks(fontsize=12)

plt.tight_layout()
plt.show()

#                           FIGURE 3a
# Set the figure size
plt.figure(figsize=(10, 6))
# Create grouped bar plot
sns.countplot(y=injury_traffic_control_devices.values, data=injury_traffic_control_devices, color='orange')
# Add labels and title
plt.xlabel('Count of Crashes Resulting in Injury', fontsize=12)
plt.ylabel('Traffic Control Devices', fontsize=12)
plt.title('Traffic Control Devices and Count of Injury Crashes', fontsize=14)
# Rotate x-axis labels for better readability
plt.xticks(fontsize=12)

# Adjust layout to prevent overlapping labels
plt.tight_layout()
# Show plot
plt.show()

#                           FIGURE 4
# Set the figure size
plt.figure(figsize=(10, 6))
# Create grouped bar plot
sns.countplot(x=df_days_temp, hue=df_imputed['crash_severity'])
# Add labels and title
plt.xlabel('Days Temperature in Farenheit', fontsize=14)
plt.ylabel('Count of Crashes', fontsize=14)
plt.title('Days Temperature and Injury Outcome', fontsize=16)
# Rotate x-axis labels for better readability
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# Set legend title and adjust legend fontsize
plt.legend(title='Crash Severity', fontsize=12, loc='upper left')

# Adjust layout to prevent overlapping labels
plt.tight_layout()
# Show plot
plt.show()