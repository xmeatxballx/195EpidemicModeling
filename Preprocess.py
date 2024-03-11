import numpy as np
import pandas as pd
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Replace 'your_data.xlsx' with the path to your Excel file
# You may need to install the 'openpyxl' or 'xlrd' package to read Excel files
df = pd.read_excel('C:\\Users\\mtpv1\\Downloads\\Data.xlsx')

# Select relevant columns (if necessary)
df = df[['Set 26']]

# Check for and handle missing values
# For simplicity, let's fill missing values with the previous value in the column
df.fillna(method='ffill', inplace=True)

window_size = 5  # Example window size

# Assuming 'df' is your DataFrame and 'Set 26' is the column with the data.

# Create lagged features for the sliding window
for i in range(1, window_size + 1):
    df[f'Set 26_lag_{i}'] = df['Set 26'].shift(i)

# Identifying columns to reverse (assuming these are the last 'window_size' columns added)
lagged_columns_reversed = df.columns[-window_size:][::-1]

# Keep the original DataFrame order, excluding the newly added lagged columns
original_columns = df.columns[:-window_size]

# Combine the original columns with the reversed lagged columns
new_column_order = original_columns.tolist() + lagged_columns_reversed.tolist()

# Reorder the DataFrame according to the new column order
df = df[new_column_order]

# Drop rows with missing values that result from the shifting operation
df.dropna(inplace=True)

df.drop('Set 26', axis=1, inplace=True)

# Now df contains your features and you can proceed with train/test split and modeling

df.to_excel('C:\\Users\\mtpv1\\Downloads\\Set 26_Processed.xlsx', index=False)