#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 19:15:21 2024

@author: SwoopSolutions
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set option to display all columns
pd.set_option('display.max_columns', None)

# Read in data from csv file and set values equal to \N ('\\N') to nan
df = pd.read_csv('chicago_2019_2022.csv', na_values=['\\N'])

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
                          'moonphase', 'lattitude', 'longitude', 'pressure',
                         'precip_type', 'feelslike', 'days_feelslikemin', 'days_feelslikemax',
                          'days_feelslike', 'days_moonphase', 'temp', 'humidity', 'snow', 'dew','precipprob', 
                          'precip', 'snowdepth', 'windspeed', 'visibility', 'cloudcover', 'conditions'
                         ])

# Reset column names to 'df_1' columns after dropping the specified columns
column_names = df_1.columns.tolist()

print('\ncolumn names after dropping unneeded columns: ', column_names)
print('\ntotal columns after dropping: ',len(column_names),'out of',total_columns,'\n')

# Print the name and data type of each attribute/column
print('uncleaned data types:\n', df_1.dtypes)

df_2 = df_1.copy()

print('\ncount of missing values in each column before cleaning:')

# Get count of missing values for each column
missing_values_count = df_2.isnull().sum()

# Display count of missing values in each column
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(missing_values_count)

# Replace nan values in specific columns with back fill method
df_2['total_injured'] = df_2['total_injured'].fillna(method='bfill')
df_2['total_killed'] = df_2['total_killed'].fillna(method='bfill')
df_2['injury_incapacitated'] = df_2['injury_incapacitated'].fillna(method='bfill')
df_2['injury_non_incapacitated'] = df_2['injury_non_incapacitated'].fillna(method='bfill')
df_2['days_snow'] = df_2['days_snow'].fillna(method='bfill')
df_2['days_snowdepth'] = df_2['days_snowdepth'].fillna(method='bfill')
df_2['days_windgust'] = df_2['days_windgust'].fillna(method='bfill')
df_2['days_preciptype'] = df_2['days_preciptype'].fillna(method='bfill')
df_2['most_severe_injury'] = df_2['most_severe_injury'].fillna('unknown')
df_2['crash_hit_and_run'] = df_2['crash_hit_and_run'].fillna('unknown')

print('\ncount of missing values in each column after cleaning:')

# Get count of missing values for each column
missing_values_count = df_2.isnull().sum()

# Display count of missing values in each column
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(missing_values_count)

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

# Replace all instances of Y or y with yes and N or n with no to make data consistent
df_2['crash_hit_and_run'] = df_2['crash_hit_and_run'].replace({'Y': 'yes', 'y': 'yes', 'N': 'no', 'n': 'no'})

# Display unique values in the crash_hit_and_run column
print('\nunique crash hit and run after cleaning: ',df_2['crash_hit_and_run'].unique(), '\n')

# Create a copy of DataFrame 'df_2' and assign it to 'df_3'
df_3 = df_2.copy()

# Change column types to category for analysis
df_3['crash_hit_and_run'] = df_3['crash_hit_and_run'].astype('category')
df_3['crash_severity'] = df_3['crash_severity'].astype('category')
df_3['days_conditions'] = df_3['days_conditions'].astype('category')
df_3['days_preciptype'] = df_3['days_preciptype'].astype('category')
df_3['most_severe_injury'] = df_3['most_severe_injury'].astype('category')
df_3['crash_type'] = df_3['crash_type'].astype('category')
df_3['contributory_cause'] = df_3['contributory_cause'].astype('category')
df_3['sec_contributory_cause'] = df_3['sec_contributory_cause'].astype('category')
df_3['traffic_control_device'] = df_3['traffic_control_device'].astype('category')
df_3['traffic_control_device_condition'] = df_3['traffic_control_device_condition'].astype('category')
df_3['road_defect'] = df_3['road_defect'].astype('category')

# Change columns to timedelta for analysis and comparison, removing '0 days' as it is not needed
df_3['crash_time'] = pd.to_timedelta(df_3['crash_time'] + ':00').astype(str).str.split().str[-1]
df_3['sunset'] = pd.to_timedelta(df_3['sunset']).astype(str).str.split().str[-1]
df_3['sunrise'] = pd.to_timedelta(df_3['sunrise']).astype(str).str.split().str[-1]

# Change columns to type int64 that should not be of type float64
df_3['total_injured'] = df_3['total_injured'].astype('int64')
df_3['total_killed'] = df_3['total_killed'].astype('int64')
df_3['injury_incapacitated'] = df_3['injury_incapacitated'].astype('int64')
df_3['injury_non_incapacitated'] = df_3['injury_non_incapacitated'].astype('int64')

print('Cleaned data types:\n', df_3.dtypes)
print('\nSample of cleaned data:\n', df_3.head(5))
print('\nShape of DataFrame df_3: ', df_3.shape, '\n')

# Create new columns and subsets of df_3 dataFrame
df_3['was_deadly'] = df_3['total_killed'] > 0
df_3['was_injury'] = df_3['total_injured'] > 0
df_3['dark_hours'] = (df_3['crash_time'] < df_3['sunrise']) | (df_3['crash_time'] > df_3['sunset'])
df_days_visibility = df_3['days_visibility'].round()
df_days_temp = (df_3['days_temp'] / 10).round() * 10
df_crash_type_injury = df_3.groupby(['crash_type', 'was_injury']).size().reset_index(name='count')
df_weather_injury = df_3.groupby(['days_conditions', 'was_injury']).size().reset_index(name='count')
df_control_condition_injury = df_3.groupby(['traffic_control_device', 'was_injury']).size().reset_index(name='count')
df_3['crash_time'] = pd.to_datetime(df_3['crash_time'])
df_rounded_crash_time = df_3['crash_time'].dt.hour.round()


# print descriptive stats for key attributes
for column in df_3:
    print(df_3[column].describe(), '\n')

# print correlation for DataFrame df_3
#print(df_3.corr(numeric_only=True))

######################   FIGURES / PLOTS  #############################

fig_labels = ['No', 'Yes']

sns.set_style('whitegrid')

#                           FIGURE 1
# Set the figure size
plt.figure(figsize=(10, 6))

# Create grouped bar plot
sns.barplot(x='count', y='crash_type', hue='was_injury', data=df_crash_type_injury)

# Add labels and title
plt.xlabel('Count of Crashes', fontsize=12)
plt.ylabel('Crash Types', fontsize=12)
plt.title('Crash Types and Injury Outcome', fontsize=14)

# Rotate x-axis labels for better readability
plt.xticks(fontsize=12)

# Set legend title and adjust legend fontsize
fig1_legend = plt.legend(title='Was_Injury', fontsize=10)

for t, l in zip(fig1_legend.texts, fig_labels):
    t.set_text(l)

# Adjust layout to prevent overlapping labels
plt.tight_layout()

# Show plot
plt.show()

#                           FIGURE 2
# Set the figure size
plt.figure(figsize=(10, 6))

# Create horizontal bar plot
sns.barplot(x='count', y='days_conditions', hue='was_injury', data=df_weather_injury)

# Add labels and title
plt.xlabel('Count of Crashes', fontsize=12)
plt.ylabel('Weather Conditions', fontsize=12)
plt.title('Weather Conditions and Injury Outcome', fontsize=14)

plt.xticks(fontsize=12)

fig2_legend = plt.legend(title='Was_Injury', fontsize=14, loc='lower right')

for t, l in zip(fig2_legend.texts, fig_labels):
    t.set_text(l)

plt.tight_layout()

plt.show()

#                           FIGURE 3
# Set the figure size
plt.figure(figsize=(10, 6))

# Create horizontal bar plot
sns.barplot(x='count', y='traffic_control_device', hue='was_injury', data=df_control_condition_injury)

# Add labels and title
plt.xlabel('Count of Crashes', fontsize=14)
plt.ylabel('Traffic Control Devices', fontsize=14)
plt.title('Traffic Control Devices and Injury Outcome', fontsize=16)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

fig3_legend = plt.legend(title='Was_Injury', fontsize=14, loc='lower right')

for t, l in zip(fig3_legend.texts, fig_labels):
    t.set_text(l)

plt.tight_layout()

plt.show()

#                           FIGURE 4
# Set the figure size
plt.figure(figsize=(10, 6))

# Create grouped bar plot
sns.countplot(x='crash_severity', hue='dark_hours', data=df_3, width=0.5, palette=['orange', 'purple'])

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
sns.countplot(x=df_days_temp, hue=df_3['was_injury'])

# Add labels and title
plt.xlabel('Days Temperature in Farenheit', fontsize=14)
plt.ylabel('Count of Crashes', fontsize=14)
plt.title('Days Temperature and Injury Outcome', fontsize=16)

# Rotate x-axis labels for better readability
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Set legend title and adjust legend fontsize
fig5_legend = plt.legend(title='Was_Injury', fontsize=12, loc='upper left')

for t, l in zip(fig5_legend.texts, fig_labels):
    t.set_text(l)

# Adjust layout to prevent overlapping labels
plt.tight_layout()

# Show plot
plt.show()

#                           FIGURE 6
# Set the figure size
plt.figure(figsize=(10, 6))

# Create grouped bar plot
sns.countplot(x=df_rounded_crash_time, hue=df_3['dark_hours'], palette=['orange', 'purple'])

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





