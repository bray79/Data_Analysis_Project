#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 19:15:21 2024

@author: SwoopSolutions
"""

import pandas as pd

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

print('\ncolumns and the count of missing values in the column before cleaning:')

# Get count of missing values for each column
missing_values_count = df_2.isnull().sum()

# Display count of missing values in each column
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(missing_values_count)

# Replace nan values in specific columns with placeholders
df_2['total_injured'] = df_2['total_injured'].fillna(-1)
df_2['total_killed'] = df_2['total_killed'].fillna(-1)
df_2['injury_incapacitated'] = df_2['injury_incapacitated'].fillna(-1)
df_2['injury_non_incapacitated'] = df_2['injury_incapacitated'].fillna(-1)
df_2['days_snow'] = df_2['days_snow'].fillna(-1)
df_2['days_snowdepth'] = df_2['days_snowdepth'].fillna(-1)
df_2['days_windgust'] = df_2['days_windgust'].fillna(-1)
df_2['days_preciptype'] = df_2['days_preciptype'].fillna('unknown')
df_2['most_severe_injury'] = df_2['most_severe_injury'].fillna('unknown')
df_2['crash_hit_and_run'] = df_2['crash_hit_and_run'].fillna('unknown')

print('\ncolumns and the count of missing values in the column after cleaning:')

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

# Change columns to type int64 that should not be of type float64
df_3['total_injured'] = df_3['total_injured'].astype('int64')
df_3['total_killed'] = df_3['total_killed'].astype('int64')
df_3['injury_incapacitated'] = df_3['injury_incapacitated'].astype('int64')
df_3['injury_non_incapacitated'] = df_3['injury_non_incapacitated'].astype('int64')

print('Cleaned data types:\n', df_3.dtypes)
print('\nSample of cleaned data:\n', df_3.head(5))
print('Shape of DataFrame df_3: ', df_3.shape)