#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 19:15:21 2024

@author: SwoopSolutions
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.impute import IterativeImputer


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

# Filter the copied DataFrame 'df_1' to include only rows where the value in the 'state' column is 'illinois'
df_1 = df_1[df_1['state'] == 'illinois']

print('\nunique states after filtering: ', df_1['state'].unique())
print('\nunique towns: ', df_1['town'].unique())

# Drop specified columns (27 cols) from DataFrame 'df_1'
df_1 = df_1.drop(columns=['country','city','state','id', 'windgust', 'winddir',
                          'moonphase', 'lattitude', 'longitude', 'pressure', 'crash_hit_and_run',
                         'precip_type', 'feelslike', 'days_feelslikemin', 'days_feelslikemax',
                          'days_feelslike', 'days_moonphase', 'temp', 'humidity', 'snow', 'dew','precipprob', 
                          'precip', 'snowdepth', 'windspeed', 'visibility', 'cloudcover', 'conditions'
                         ])

# Reset column names to 'df_1' columns after dropping the specified columns
column_names = df_1.columns.tolist()

print('\ncolumn names after dropping unneeded columns: ', column_names)
print('\ntotal columns after dropping: ',len(column_names),'out of',total_columns,'\n')

# Print the name and data type of each attribute/column
#print('uncleaned data types:\n', df_1.dtypes)

df_2 = df_1.copy()

print('\ncount of missing values in each column before cleaning:')

# Get count of missing values for each column
missing_values_count = df_2.isnull().sum()

# Display count of missing values in each column
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(missing_values_count)
    
df_2['crash_date'] = pd.to_datetime(df_2['crash_date'], format='%m/%d/%y')
# Change columns to timedelta for analysis and comparison, removing '0 days' as it is not needed
df_2['crash_time'] = pd.to_timedelta(df_2['crash_time'] + ':00').astype(str).str.split().str[-1]
df_2['sunset'] = pd.to_timedelta(df_2['sunset']).astype(str).str.split().str[-1]
df_2['sunrise'] = pd.to_timedelta(df_2['sunrise']).astype(str).str.split().str[-1]

df_2['town'] = df_2['town'].astype('category')
df_2['crash_severity'] = df_2['crash_severity'].astype('category')
df_2['days_conditions'] = df_2['days_conditions'].astype('category')
df_2['days_preciptype'] = df_2['days_preciptype'].astype('category')
df_2['most_severe_injury'] = df_2['most_severe_injury'].astype('category')
df_2['crash_type'] = df_2['crash_type'].astype('category')
df_2['contributory_cause'] = df_2['contributory_cause'].astype('category')
df_2['sec_contributory_cause'] = df_2['sec_contributory_cause'].astype('category')
df_2['traffic_control_device'] = df_2['traffic_control_device'].astype('category')
df_2['traffic_control_device_condition'] = df_2['traffic_control_device_condition'].astype('category')
df_2['road_defect'] = df_2['road_defect'].astype('category')

print(df_2.info())

numeric_column_names = df_2.select_dtypes(include=['int64', 'float64']).columns
categorical_column_names = df_2.select_dtypes(include='category').columns
datetime_column_names = df_2.select_dtypes(include=['datetime64[ns]', 'timedelta64[ns]', 'object']).columns

df_numeric = df_2[numeric_column_names].copy()
df_categorical = df_2[categorical_column_names].copy()
df_datetime = df_2[datetime_column_names].copy()

imp_numeric = IterativeImputer(estimator=LinearRegression(), max_iter=3, tol=1e-10, imputation_order='roman')
df_numeric_imputed = imp_numeric.fit_transform(df_numeric)
df_numeric_imputed = pd.DataFrame(df_numeric_imputed, columns=numeric_column_names)

df_numeric_imputed['total_injured'] = df_numeric_imputed['total_injured'].astype('int64')
df_numeric_imputed['total_killed'] = df_numeric_imputed['total_killed'].astype('int64')
df_numeric_imputed['injury_incapacitated'] = df_numeric_imputed['injury_incapacitated'].astype('int64')
df_numeric_imputed['injury_non_incapacitated'] = df_numeric_imputed['injury_non_incapacitated'].astype('int64')
df_numeric_imputed['num_vehicles_in_crash'] = df_numeric_imputed['num_vehicles_in_crash'].astype('int64')
df_numeric_imputed['days_precipprob'] = df_numeric_imputed['days_precipprob'].astype('int64')
df_numeric_imputed['days_uvindex'] = df_numeric_imputed['days_uvindex'].astype('int64')

for col in df_categorical:
    df_categorical[col].fillna(df_categorical[col].mode()[0], inplace=True)

df_numeric_imputed.reset_index(drop=True, inplace=True)
df_categorical.reset_index(drop=True, inplace=True)
df_datetime.reset_index(drop=True, inplace=True)

df_imputed = pd.concat([df_numeric_imputed, df_categorical, df_datetime], axis=1)

print('\ncount of missing values in each column after cleaning:')

# Get count of missing values for each column
missing_values_count = df_imputed.isnull().sum()

# Display count of missing values in each column
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(missing_values_count)
    
print('\nCleaned data types:\n', df_imputed.dtypes)
print('\nSample of cleaned data:\n', df_imputed.head(5))
print('\nShape of DataFrame df_imputed: ', df_imputed.shape, '\n')

# Create new columns and subsets of df_3 dataFrame
df_imputed['was_deadly'] = df_imputed['total_killed'] > 0
df_imputed['was_injury'] = df_imputed['total_injured'] > 0
df_imputed['dark_hours'] = (df_imputed['crash_time'] < df_imputed['sunrise']) | (df_imputed['crash_time'] > df_imputed['sunset'])
df_days_visibility = df_imputed['days_visibility'].round()
df_days_temp = (df_imputed['days_temp'] / 10).round() * 10
df_crash_type_injury = df_imputed.groupby(['crash_type', 'was_injury']).size().reset_index(name='count')
df_weather_injury = df_imputed.groupby(['days_conditions', 'was_injury']).size().reset_index(name='count')
df_control_condition_injury = df_imputed.groupby(['traffic_control_device', 'was_injury']).size().reset_index(name='count')
df_imputed['crash_time'] = pd.to_datetime(df_imputed['crash_time'])
df_rounded_crash_time = df_imputed['crash_time'].dt.hour.round()
df_imputed['rounded_crash_time'] = df_imputed['crash_time'].dt.hour.round()

injury_crashes_types = df_imputed.loc[df_imputed['was_injury'] == True, 'crash_type']
deadly_crashes_types = df_imputed.loc[df_imputed['was_deadly'] == True, 'crash_type']

injury_days_conditions_types = df_imputed.loc[df_imputed['was_injury'] == True, 'days_conditions']
deadly_days_conditions_types = df_imputed.loc[df_imputed['was_deadly'] == True, 'days_conditions']

injury_traffic_control_devices = df_imputed.loc[df_imputed['was_injury'] == True, 'traffic_control_device']


# print descriptive stats for key attributes
for column in df_imputed:
    print(df_imputed[column].describe(), '\n')


######################   FIGURES / PLOTS  #############################

fig_labels = ['No', 'Yes']

sns.set_style('whitegrid')

#                           FIGURE 1
# Set the figure size
plt.figure(figsize=(10, 6))

# Create grouped bar plot
sns.countplot(y='crash_type', data=df_imputed, color='blue')

# Add labels and title
plt.xlabel('Count of Crashes', fontsize=12)
plt.ylabel('Crash Types', fontsize=12)
plt.title('Crash Types and Count of Crashes', fontsize=14)

# Rotate x-axis labels for better readability
plt.xticks(fontsize=12)

# Adjust layout to prevent overlapping labels
plt.tight_layout()

# Show plot
plt.show()

#                           FIGURE 1a
# Set the figure size
plt.figure(figsize=(10, 6))

# Create grouped bar plot
sns.countplot(y=injury_crashes_types.values, data=injury_crashes_types, color='orange')

# Add labels and title
plt.xlabel('Count of Crashes Resulting in Injury', fontsize=12)
plt.ylabel('Crash Types', fontsize=12)
plt.title('Crash Types and Count of Injury Crashes', fontsize=14)

# Rotate x-axis labels for better readability
plt.xticks(fontsize=12)

# Adjust layout to prevent overlapping labels
plt.tight_layout()

# Show plot
plt.show()

#                           FIGURE 1b
# Set the figure size
plt.figure(figsize=(10, 6))

# Create grouped bar plot
sns.countplot(y=deadly_crashes_types.values, data=deadly_crashes_types, color='red')

# Add labels and title
plt.xlabel('Count of Crashes Resulting in Death', fontsize=12)
plt.ylabel('Crash Types', fontsize=12)
plt.title('Crash Types and Count of Deadly Crashes', fontsize=14)

# Rotate x-axis labels for better readability
plt.xticks(fontsize=12)

# Adjust layout to prevent overlapping labels
plt.tight_layout()

# Show plot
plt.show()

#                           FIGURE 2
# Set the figure size
plt.figure(figsize=(10, 6))

# Create horizontal bar plot
sns.countplot(y='days_conditions', data=df_imputed, color='blue')

# Add labels and title
plt.xlabel('Count of Crashes', fontsize=12)
plt.ylabel('Weather Conditions', fontsize=12)
plt.title('Weather Conditions and Count of Crashes', fontsize=14)

plt.xticks(fontsize=12)

plt.tight_layout()

plt.show()

#                           FIGURE 2a
# Set the figure size
plt.figure(figsize=(10, 6))

# Create horizontal bar plot
sns.countplot(y=injury_days_conditions_types.values, data=injury_days_conditions_types, color='orange')

# Add labels and title
plt.xlabel('Count of Crashes Resulting in Injury', fontsize=12)
plt.ylabel('Weather Conditions', fontsize=12)
plt.title('Weather Conditions and Count of Injury Crashes', fontsize=14)

plt.xticks(fontsize=12)

plt.tight_layout()

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
sns.countplot(x='crash_severity', hue='dark_hours', data=df_imputed, width=0.5, palette=['orange', 'purple'])

# Add labels and title
plt.xlabel('Crash Severity', fontsize=12)
plt.ylabel('Count of Crashes', fontsize=12)
plt.title('Crash Severity and Darkness', fontsize=14)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

fig4_legend = plt.legend(title='Was_Dark', fontsize=12, loc='upper right')

for t, l in zip(fig4_legend.texts, fig_labels):
    t.set_text(l)

plt.tight_layout()

plt.show()

#                           FIGURE 5
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

#                           FIGURE 6
# Set the figure size
plt.figure(figsize=(10, 6))

# Create grouped bar plot
sns.countplot(x=df_rounded_crash_time, hue=df_imputed['dark_hours'], palette=['orange', 'purple'])

# Add labels and title
plt.xlabel('Crash Times Rounded', fontsize=14)
plt.ylabel('Count of Crashes', fontsize=14)
plt.title('Crash Times and Darkness', fontsize=16)

# Rotate x-axis labels for better readability
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Set legend title and adjust legend fontsize
fig6_legend = plt.legend(title='Was_Dark', fontsize=12, loc='upper left')

for t, l in zip(fig6_legend.texts, fig_labels):
    t.set_text(l)

# Adjust layout to prevent overlapping labels
plt.tight_layout()

# Show plot
plt.show()

"""

# Display columms to check for inconsistencies (particulary columns with type object)
print('\nunique total injured: ', sorted(df_2['total_injured'].unique()))
print('\nunique total killed: ', df_2['total_killed'].unique())
print('\nunique most severe injuries: ', df_2['most_severe_injury'].unique())
print('\nunique crash types: ', df_2['crash_type'].unique())
print('\nunique contributory causes: ', df_2['contributory_cause'].unique())
print('\nunique crash severity: ', df_2['crash_severity'].unique())
print('\nunique traffic control devices: ', df_2['traffic_control_device'].unique())
print('\nunique traffic control device conditions: ', df_2['traffic_control_device_condition'].unique())
print('\nunique road defects: ', df_2['road_defect'].unique())
print('\nunique days precip types: ', df_2['days_preciptype'].unique())
print('\nunique days conditions: ', df_2['days_conditions'].unique())
print('\nunique crash hit and run before cleaning: ', df_2['crash_hit_and_run'].unique())

"""

