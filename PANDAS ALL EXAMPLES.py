#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Handling Missing Data:

df.isnull(): Check for missing values in the DataFrame.
df.dropna(): Drop rows with missing values.
df.fillna(value): Fill missing values with a specified value.
df.interpolate(): Interpolate missing values based on neighboring values.


# In[1]:


import pandas as pd

data = {'A': [1, 2, None, 4],
        'B': [5, None, 7, 8]}

df = pd.DataFrame(data)

# Check for missing values in the DataFrame
missing_values = df.isnull()
print(missing_values)


# In[2]:


import pandas as pd

data = {'A': [1, 2, None, 4],
        'B': [5, None, 7, 8]}

df = pd.DataFrame(data)

# Drop rows with missing values
df_dropped = df.dropna()
print(df_dropped)


# In[3]:


import pandas as pd

data = {'A': [1, 2, None, 4],
        'B': [5, None, 7, 8]}

df = pd.DataFrame(data)

# Fill missing values with a specified value
df_filled = df.fillna(0)
print(df_filled)


# In[4]:


import pandas as pd

data = {'A': [1, 2, None, 4],
        'B': [5, None, 7, 8]}

df = pd.DataFrame(data)

# Interpolate missing values based on neighboring values
df_interpolated = df.interpolate()
print(df_interpolated)


# In[ ]:


Removing Duplicates:

df.duplicated(): Check for duplicate rows.
df.drop_duplicates(): Remove duplicate rows.


# In[6]:


import pandas as pd

data = {'A': [1, 2, 2, 3, 4],
        'B': ['x', 'y', 'y', 'z', 'z']}

df = pd.DataFrame(data)

# Check for duplicate rows in the DataFrame
duplicates = df.duplicated()
print(duplicates)
print(df)


# In[7]:


import pandas as pd

data = {'A': [1, 2, 2, 3, 4],
        'B': ['x', 'y', 'y', 'z', 'z']}

df = pd.DataFrame(data)

# Remove duplicate rows from the DataFrame
df_no_duplicates = df.drop_duplicates()
print(df_no_duplicates)


# In[ ]:


#Data Transformation:

#df.rename(columns={'old_name': 'new_name'}): Rename columns.
#df.drop(columns=['col_name']): Drop columns.
#df.replace(old_value, new_value): Replace specific values.
#df.apply(func): Apply a function to rows or columns.
#df.groupby('col_name').agg(func): Group data and apply 


# In[8]:


import pandas as pd

data = {'Old_Column': [1, 2, 3],
        'Another_Column': ['x', 'y', 'z']}

df = pd.DataFrame(data)

# Rename columns
df_renamed = df.rename(columns={'Old_Column': 'New_Column', 'Another_Column': 'New_Column2'})
print(df_renamed)


# In[9]:


import pandas as pd

data = {'A': [1, 2, 3],
        'B': ['x', 'y', 'z']}

df = pd.DataFrame(data)

# Drop columns
df_dropped = df.drop(columns=['B'])
print(df_dropped)


# In[10]:


import pandas as pd

data = {'A': [1, 2, 3, 4, 5],
        'B': ['x', 'y', 'z', 'y', 'x']}

df = pd.DataFrame(data)

# Replace specific values in a column
df_replaced = df.replace('x', 'replacement')
print(df_replaced)


# In[11]:


import pandas as pd

data = {'A': [1, 2, 3],
        'B': [4, 5, 6]}

df = pd.DataFrame(data)

# Apply a function to a column
def square(x):
    return x ** 2

df['B_squared'] = df['B'].apply(square)
print(df)


# In[12]:


import pandas as pd

data = {'Category': ['A', 'B', 'A', 'B', 'A'],
        'Value': [10, 20, 15, 25, 12]}

df = pd.DataFrame(data)

# Group data by 'Category' and calculate the sum of 'Value' for each group
grouped_df = df.groupby('Category').agg(sum)
print(grouped_df)


# In[13]:


#Data Type Conversion:

#df.astype({'col_name': 'new_dtype'}): Convert data types of specific columns.
#pd.to_datetime(df['date_col']): Convert a column to datetime format.


# In[14]:


import pandas as pd

data = {'A': [1, 2, 3],
        'B': ['x', 'y', 'z']}

df = pd.DataFrame(data)

# Convert data types of specific columns
df_converted = df.astype({'A': float})
print(df_converted.dtypes)


# In[15]:


import pandas as pd

data = {'Date': ['2022-01-01', '2022-01-02', '2022-01-03'],
        'Value': [10, 20, 30]}

df = pd.DataFrame(data)

# Convert a column to datetime format
df['Date'] = pd.to_datetime(df['Date'])
print(df.dtypes)


# In[16]:


#String Cleaning:

#df['col_name'].str.replace(old_str, new_str): Replace substrings in string columns.
#df['col_name'].str.strip(): Remove leading and trailing whitespace.
#df['col_name'].str.lower(): Convert strings to lowercase


# In[17]:


import pandas as pd

data = {'Name': ['John Smith', 'Jane Doe', 'Alice Johnson'],
        'City': ['New York', 'Los Angeles', 'Chicago']}

df = pd.DataFrame(data)

# Replace substrings in string columns
df['City'] = df['City'].str.replace(' ', '_')
print(df)


# In[18]:


import pandas as pd

data = {'Name': ['  John Smith  ', '  Jane Doe  ', '  Alice Johnson  ']}

df = pd.DataFrame(data)

# Remove leading and trailing whitespace
df['Name'] = df['Name'].str.strip()
print(df)


# In[19]:


import pandas as pd

data = {'Name': ['John Smith', 'Jane Doe', 'Alice Johnson']}

df = pd.DataFrame(data)

# Convert strings to lowercase
df['Name'] = df['Name'].str.lower()
print(df)


# In[ ]:


#Handling Outliers:

#df['col_name'].clip(lower, upper): Clip outliers within a specified range.
#df['col_name'].apply(lambda x: x if x < upper else np.nan): Replace outliers with NaN.


# In[20]:


import pandas as pd

data = {'Score': [90, 110, 95, 120, 85]}

df = pd.DataFrame(data)

# Clip outliers within a specified range
df['Score_clipped'] = df['Score'].clip(lower=100, upper=110)
print(df)


# In[21]:


import pandas as pd
import numpy as np

data = {'Score': [90, 110, 95, 120, 85]}

df = pd.DataFrame(data)

# Replace outliers with NaN
upper_threshold = 100
df['Score_replaced'] = df['Score'].apply(lambda x: x if x < upper_threshold else np.nan)
print(df)


# In[ ]:


#Reshaping Data:

#pd.melt(df, id_vars=['col1'], value_vars=['col2']): Convert wide data to long format.
#df.pivot_table(index='index_col', columns='col_name', values='value_col'): Create a pivot table.


# In[22]:


import pandas as pd

data = {'Name': ['John', 'Jane', 'Alice'],
        'Math': [90, 85, 95],
        'Science': [88, 92, 90]}

df = pd.DataFrame(data)

# Convert wide data to long format using pd.melt()
df_melted = pd.melt(df, id_vars=['Name'], value_vars=['Math', 'Science'], var_name='Subject', value_name='Score')
print(df_melted)


# In[23]:


import pandas as pd

data = {'Name': ['John', 'Jane', 'Alice', 'John', 'Jane', 'Alice'],
        'Subject': ['Math', 'Math', 'Math', 'Science', 'Science', 'Science'],
        'Score': [90, 85, 95, 88, 92, 90]}

df = pd.DataFrame(data)

# Create a pivot table
pivot_table = df.pivot_table(index='Name', columns='Subject', values='Score', aggfunc='mean')
print(pivot_table)


# In[ ]:


#Data Validation:

#df['col_name'].str.isnumeric(): Check if values are numeric.
#df['col_name'].isin(['value1', 'value2']): Check if values are in a specific list.


# In[24]:


import pandas as pd

data = {'Age': ['25', '30', '40', 'invalid', '28']}

df = pd.DataFrame(data)

# Check if values in 'Age' column are numeric
df['Is_Numeric'] = df['Age'].str.isnumeric()
print(df)


# In[25]:


import pandas as pd

data = {'Category': ['A', 'B', 'C', 'B', 'A']}

df = pd.DataFrame(data)

# Check if values in 'Category' column are in a specific list
valid_categories = ['A', 'B']
df['Is_Valid_Category'] = df['Category'].isin(valid_categories)
print(df)


# In[ ]:


#Datetime Handling:

#df['date_col'] = pd.to_datetime(df['date_col']): Convert a column to datetime format.
#df['date_col'].dt.year: Extract year from datetime column.
#df['date_col'].dt.month: Extract month from datetime column.


# In[26]:


import pandas as pd

data = {'Date': ['2022-01-01', '2022-01-02', '2022-01-03'],
        'Value': [10, 20, 30]}

df = pd.DataFrame(data)

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])
print(df.dtypes)


# In[27]:


import pandas as pd

data = {'Date': ['2022-01-01', '2022-02-15', '2022-03-20'],
        'Value': [10, 20, 30]}

df = pd.DataFrame(data)

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Extract year and month from 'Date' column
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
print(df)


# In[28]:


import pandas as pd

data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 22, 28]}

df = pd.DataFrame(data)

# Sort DataFrame by 'Age' in descending order
df_sorted = df.sort_values(by='Age', ascending=False)
print(df_sorted)


# In[29]:


import pandas as pd

data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 22, 28]}

df = pd.DataFrame(data, index=[2, 1, 0])  # Custom index

# Sort DataFrame by index
df_sorted = df.sort_index()
print(df_sorted)


# In[ ]:


#Grouping and Aggregating Data:

#df.groupby('col_name').mean(): Calculate mean for each group.
#df.groupby('col_name').sum(): Calculate sum for each group.
#df.groupby('col_name').size(): Count occurrences in each group.
#df.groupby('col_name').agg({'col1': 'mean', 'col2': 'sum'}): Apply multiple aggregation functions to different 


# In[30]:


import pandas as pd

data = {'Category': ['A', 'B', 'A', 'B'],
        'Value': [10, 20, 15, 25]}

df = pd.DataFrame(data)

# Group by 'Category' and calculate the mean value for each group
grouped_mean = df.groupby('Category').mean()
print(grouped_mean)


# In[31]:


import pandas as pd

data = {'Category': ['A', 'B', 'A', 'B'],
        'Value': [10, 20, 15, 25]}

df = pd.DataFrame(data)

# Group by 'Category' and calculate the sum of values for each group
grouped_sum = df.groupby('Category').sum()
print(grouped_sum)


# In[32]:


import pandas as pd

data = {'Category': ['A', 'B', 'A', 'B'],
        'Value': [10, 20, 15, 25]}

df = pd.DataFrame(data)

# Group by 'Category' and count occurrences in each group
grouped_count = df.groupby('Category').size()
print(grouped_count)


# In[33]:


import pandas as pd

data = {'Category': ['A', 'B', 'A', 'B'],
        'Value1': [10, 20, 15, 25],
        'Value2': [5, 10, 8, 12]}

df = pd.DataFrame(data)

# Group by 'Category' and calculate mean for 'Value1' and sum for 'Value2'
grouped_agg = df.groupby('Category').agg({'Value1': 'mean', 'Value2': 'sum'})
print(grouped_agg)


# In[ ]:


#String Manipulation:

#df['col_name'].str.len(): Calculate the length of strings in a column.
#df['col_name'].str.contains('substring'): Check if a substring is present in each value.


# In[34]:


import pandas as pd

data = {'Name': ['Alice', 'Bob', 'Charlie']}

df = pd.DataFrame(data)

# Calculate the length of strings in the 'Name' column
df['Name_Length'] = df['Name'].str.len()
print(df)


# In[35]:


import pandas as pd

data = {'Title': ['Data Science', 'Machine Learning', 'Python Programming']}

df = pd.DataFrame(data)

# Check if 'Science' is present in each value of the 'Title' column
df['Contains_Science'] = df['Title'].str.contains('Science')
print(df)


# In[ ]:


#Date and Time Manipulation:

#df['date_col'].dt.dayofweek: Extract day of the week from a datetime column.
#df['date_col'].dt.day_name(): Get the name of the day from a datetime column.


# In[36]:


import pandas as pd

data = {'Date': ['2023-08-01', '2023-08-02', '2023-08-03']}

df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])

# Extract the day of the week (as an integer) from the 'Date' column
df['Day_of_Week'] = df['Date'].dt.dayofweek
print(df)


# In[37]:


import pandas as pd

data = {'Date': ['2023-08-01', '2023-08-02', '2023-08-03']}

df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])

# Extract the day of the week (as an integer) from the 'Date' column
df['Day_of_Week'] = df['Date'].dt.dayofweek
print(df)


# In[39]:


import pandas as pd

data = {'Date': ['2023-08-01', '2023-08-02', '2023-08-03']}

df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])

# Get the name of the day of the week from the 'Date' column
df['Day_Name'] = df['Date'].dt.day_name()
print(df)


# In[ ]:


#Reshaping Data:

#pd.concat([df1, df2], axis=0): Concatenate DataFrames vertically.
#pd.concat([df1, df2], axis=1): Concatenate DataFrames horizontally.


# In[40]:


import pandas as pd

data1 = {'Name': ['Alice', 'Bob'],
         'Age': [25, 30]}

data2 = {'Name': ['Charlie', 'David'],
         'Age': [28, 22]}

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# Concatenate DataFrames vertically
result = pd.concat([df1, df2], axis=0)
print(result)


# In[41]:


import pandas as pd

data1 = {'Name': ['Alice', 'Bob'],
         'Age': [25, 30]}

data2 = {'Country': ['USA', 'Canada'],
         'Gender': ['F', 'M']}

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# Concatenate DataFrames horizontally
result = pd.concat([df1, df2], axis=1)
print(result)


# In[ ]:


#Merge and Join Data:

#pd.merge(df1, df2, on='key_col'): Merge two DataFrames based on a common column.
#df1.join(df2, on='key_col'): Join two DataFrames based 


# In[42]:


import pandas as pd

data1 = {'ID': [1, 2, 3],
         'Name': ['Alice', 'Bob', 'Charlie']}

data2 = {'ID': [2, 3, 4],
         'Age': [25, 30, 28]}

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# Merge DataFrames based on the 'ID' column
result = pd.merge(df1, df2, on='ID')
print(result)


# In[43]:


import pandas as pd

data1 = {'ID': [1, 2, 3],
         'Name': ['Alice', 'Bob', 'Charlie']}

data2 = {'Age': [25, 30, 28]}

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2, index=[2, 0, 1])  # Index matches the 'ID' column

# Join DataFrames based on the index
result = df1.join(df2, on='ID')
print(result)


# In[ ]:


#Pivot Tables and Crosstabs:

#pd.pivot_table(df, values='value_col', index='index_col', columns='col_name', aggfunc='mean'): Create a pivot table.
#pd.crosstab(df['col1'], df['col2']): Create a cross-tabulation table.


# In[44]:


import pandas as pd

data = {'Date': ['2023-08-01', '2023-08-01', '2023-08-02', '2023-08-02'],
        'Product': ['A', 'B', 'A', 'B'],
        'Sales': [100, 200, 150, 250]}

df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])

# Create a pivot table with 'Date' as index, 'Product' as columns, and 'Sales' as values
pivot_table = pd.pivot_table(df, values='Sales', index='Date', columns='Product', aggfunc='sum')
print(pivot_table)


# In[45]:


import pandas as pd

data = {'Category': ['A', 'B', 'A', 'B', 'A', 'B'],
        'Status': ['Pass', 'Fail', 'Fail', 'Pass', 'Pass', 'Fail']}

df = pd.DataFrame(data)

# Create a cross-tabulation table for 'Category' and 'Status'
crosstab = pd.crosstab(df['Category'], df['Status'])
print(crosstab)


# In[46]:


import pandas as pd

data = {'Year': [2018, 2019, 2020, 2021],
        'Revenue': [1000, 1500, 1200, 1800]}

df = pd.DataFrame(data)

# Create a line plot
df.plot(kind='line', x='Year', y='Revenue')

# Create a bar plot
df.plot(kind='bar', x='Year', y='Revenue')

# Create a histogram
df.plot(kind='hist', y='Revenue', bins=3)


# In[47]:


import pandas as pd

data = {'Height': [160, 170, 165, 175, 155],
        'Weight': [60, 70, 65, 80, 50]}

df = pd.DataFrame(data)

# Create a scatter plot
df.plot.scatter(x='Height', y='Weight')


# In[ ]:


#resample(): Resample time series data.
#shift(): Shift data in a time series.
#rolling(): Compute rolling statistics for time series data.


# In[54]:


import pandas as pd

# Create a DataFrame with daily stock prices
data = {'Date': pd.date_range(start='2023-01-01', periods=365),
        'Price': range(365)}

df = pd.DataFrame(data)
df.set_index('Date', inplace=True)

# Resample to monthly frequency and calculate the mean
monthly_mean = df.resample('M').mean()
monthly_mean
#resample(): Resampling involves changing the frequency of your time
#series data. It's commonly used to aggregate data over different time intervals (e.g., daily to monthly).


# In[52]:


#shift(): This function allows you to shift the data values in a time series by a specified number of periods.


import pandas as pd

# Create a DataFrame with monthly sales data
data = {'Month': pd.date_range(start='2023-01-01', periods=12, freq='M'),
        'Sales': [100, 150, 120, 130, 110, 180, 200, 170, 140, 160, 190, 220]}

df = pd.DataFrame(data)
df.set_index('Month', inplace=True)

# Shift the sales data by one month
df['Shifted_Sales'] = df['Sales'].shift(1)
df


# In[53]:


#rolling(): The rolling() function computes rolling statistics (such as mean, median, etc.) 
#over a specified window of time in a time series.
import pandas as pd

# Create a DataFrame with daily stock prices
data = {'Date': pd.date_range(start='2023-01-01', periods=30),
        'Price': [100, 110, 120, 125, 130, 125, 135, 140, 150, 155,
                  160, 165, 160, 170, 175, 180, 185, 190, 195, 200,
                  210, 215, 220, 225, 230, 235, 240, 245, 250, 255]}

df = pd.DataFrame(data)
df.set_index('Date', inplace=True)

# Calculate the rolling mean over a window of 5 days
df['Rolling_Mean'] = df['Price'].rolling(window=5).mean()
df


# In[56]:


import pandas as pd

data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
df = pd.DataFrame(data, index=['row1', 'row2', 'row3'])

# Select specific rows by labels
selected_rows = df.loc[['row1', 'row3']]

# Select specific columns by labels
selected_columns = df.loc[:, ['A']]
selected_columns


# In[62]:


import pandas as pd

data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
df = pd.DataFrame(data)

# Select the first row by its position
first_row = df.iloc[0]

# Select specific rows by positions
selected_rows = df.iloc[[0, 2]]
selected_rows
# Select specific columns by positions
selected_columns = df.iloc[:, [1]]

selected_columns


# In[63]:


import pandas as pd

# Create a sample DataFrame
data = {'Date': pd.date_range(start='2023-01-01', periods=5, freq='D'),
        'Value': [10, 20, 30, 40, 50]}
df = pd.DataFrame(data)

# Shift the 'Value' column by 1 position
df['Shifted'] = df['Value'].shift(1)

print(df)


# In[64]:


import pandas as pd

# Create a sample DataFrame
data = {'Date': pd.date_range(start='2023-01-01', periods=5, freq='D'),
        'Value': [10, 20, 30, 40, 50]}
df = pd.DataFrame(data)

# Calculate the differences between consecutive values
df['Diff'] = df['Value'].diff()

print(df)


# In[ ]:


#In this example, the diff() function calculates the differences between 
#consecutive values of the 'Value' column. The first row has a NaN (Not a Number) 
#value in the 'Diff' column since there is no previous value to calculate the difference from. 
#The subsequent rows show the differences between consecutive values.







# In[ ]:


#pct_change() function is used to calculate the percentage
#change between consecutive values of an existing column and
#create a new column with those percentage changes. 


# In[65]:


import pandas as pd

# Create a sample DataFrame
data = {'Date': pd.date_range(start='2023-01-01', periods=5, freq='D'),
        'Value': [10, 20, 30, 40, 50]}
df = pd.DataFrame(data)

# Calculate the percentage change between consecutive values
df['Pct_Change'] = df['Value'].pct_change() * 100

print(df)


# In[ ]:


#pct_change() function calculates the percentage change between consecutive values of the 
#'Value' column. The first row has a NaN (Not a Number) value in the 'Pct_Change' 
#column since there is no previous value to calculate the percentage change from. 
#The subsequent rows show the percentage changes between consecutive values, multiplied by 100 to represent percentages.


# In[ ]:


#rolling() function is used to calculate the rolling mean of an existing 
#column over a specified window size and create a new column with those rolling mean values.


# In[66]:


import pandas as pd

# Create a sample DataFrame
data = {'Date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
        'Value': [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]}
df = pd.DataFrame(data)

# Calculate the rolling mean with a window size of 7
df['Rolling_Mean'] = df['Value'].rolling(window=7).mean()

print(df)


# In[ ]:


#rolling(window=7).mean() function calculates the rolling mean of the 
#'Value' column over a window size of 7 days. The first 6 rows have NaN (Not a Number)
#values in the 'Rolling_Mean' column since there are not enough previous values to calculate the 
#rolling mean. Starting from the 7th row, the rolling mean is calculated using the preceding 7 values.


# In[ ]:


#resample() function is used to resample time series data at a
#specified frequency and apply a aggregation function (in this case, the mean) 
#to the data within each time interval. 


# In[68]:


import pandas as pd

# Create a sample DataFrame with a DatetimeIndex
data = {'Date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
        'Value': [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]}
df = pd.DataFrame(data)
df.set_index('Date', inplace=True)  # Set 'Date' column as the index

# Now you can use resample
df_resampled = df.resample('W').mean()

print(df_resampled)



# In[ ]:


# resample('W').mean() function resamples the 'Date' index of the DataFrame at a weekly frequency
#('W') and calculates the mean of the 'Value' column within each week. The resulting df_resampled 
#DataFrame shows the mean value for each week.

#Resampling is a powerful technique in time series analysis that allows you 
#to aggregate data over different time intervals, which can be useful for creating summaries,
#identifying patterns, and visualizing trends in your data.


# In[ ]:


#cumulative sum of the 'Value' column in your DataFrame and creating a new column called 
#'Cumulative_Sum' to store the results. This can be very useful for tracking the cumulative 
#total of a particular value or metric over time.


# In[69]:


import pandas as pd

# Create a sample DataFrame
data = {'Date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
        'Value': [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]}
df = pd.DataFrame(data)

# Calculate the cumulative sum of the 'Value' column
df['Cumulative_Sum'] = df['Value'].cumsum()

print(df)


# In[ ]:


#This information can be valuable for understanding the accumulation
#of values over time or for creating cumulative charts and visualizations


# In[ ]:


#Cumulative_Product' column keeps track of the cumulative product 
#of the 'Value' column as you move down the DataFrame. 
#This can be helpful for understanding the cumulative effect of multiplicative changes in a value over time


# In[70]:


import pandas as pd

# Create a sample DataFrame
data = {'Date': pd.date_range(start='2023-01-01', periods=5, freq='D'),
        'Value': [2, 3, 4, 5, 6]}
df = pd.DataFrame(data)

# Calculate the cumulative product of the 'Value' column
df['Cumulative_Product'] = df['Value'].cumprod()

print(df)


# In[ ]:


#calculating the rolling standard deviation of the 'Value'
#column in your DataFrame using a window size of 30, and then creating
#a new column called 'Rolling_Std' to store the calculated values. Rolling statistics
#like rolling mean and rolling standard deviation can help you identify trends and variations
#in your data over a specific window of observations.


# In[71]:


import pandas as pd
import numpy as np

# Create a sample DataFrame
data = {'Date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
        'Value': np.random.randn(100)}  # Generating random values
df = pd.DataFrame(data)

# Calculate the rolling standard deviation of the 'Value' column
df['Rolling_Std'] = df['Value'].rolling(window=30).std()

print(df)


# In[ ]:


#In this example, the 'Rolling_Std' column contains the rolling standard deviation 
#of the 'Value' column with a window size of 30. As you move down the DataFrame,
#the rolling standard deviation is calculated for each window of 30 observations.
#This can help you identify periods of high or low variability in the data and is 
#often used for smoothing noisy time series data.


# In[ ]:


# you're calculating the exponentially weighted moving average (EWMA) 
#of the 'Value' column in your DataFrame using a span of 10,
#and then creating a new column called 'EWMA' to store the calculated values.
#The EWMA is a type of moving average that gives more weight to recent observations,
#making it responsive to changes in the data.


# In[72]:


import pandas as pd
import numpy as np

# Create a sample DataFrame
data = {'Date': pd.date_range(start='2023-01-01', periods=50, freq='D'),
        'Value': np.random.randn(50)}  # Generating random values
df = pd.DataFrame(data)

# Calculate the exponentially weighted moving average (EWMA) of the 'Value' column
df['EWMA'] = df['Value'].ewm(span=10).mean()

print(df)


# In[ ]:


# 'EWMA' column contains the exponentially weighted moving average
#of the 'Value' column with a span of 10. The EWMA provides a smoothed
#representation of the data that is sensitive to recent changes, making 
#it useful for detecting trends and patterns in noisy time series data


# In[73]:


read_csv(): Read a CSV file into a DataFrame.
head(): Display the first n rows of a DataFrame.
tail(): Display the last n rows of a DataFrame.
info(): Display a summary of a DataFrame's structure.
describe(): Generate descriptive statistics of a DataFrame.
shape: Return the dimensions (rows and columns) of a DataFrame.
columns: Access the column labels of a DataFrame.
dtypes: Display the data types of columns in a DataFrame.
isnull(): Check for missing values in a DataFrame.
dropna(): Drop rows with missing values.
fillna(): Fill missing values with a specified value or method.
groupby(): Group data based on a specified column.
agg(): Apply aggregation functions to grouped data.
sort_values(): Sort a DataFrame based on one or more columns.
apply(): Apply a function to rows or columns of a DataFrame.
pivot_table(): Create a pivot table from data.
merge(): Merge two DataFrames based on a common column.
concat(): Concatenate DataFrames along rows or columns.
value_counts(): Count occurrences of unique values in a Series.
crosstab(): Create a cross-tabulation table.
unique(): Get unique values from a Series.
nunique(): Count the number of unique values in a Series.
map(): Apply a mapping function to a Series.
str.replace(): Replace substrings in string columns.
str.extract(): Extract substrings from string columns.
str.len(): Calculate the length of strings in a Series.
str.lower(): Convert strings to lowercase.
str.upper(): Convert strings to uppercase.
str.strip(): Remove leading and trailing whitespace from strings.
str.contains(): Check if a substring is present in each value.
astype(): Convert data types of columns in a DataFrame.
to_datetime(): Convert a column to datetime format.
drop(): Drop specified columns or rows from a DataFrame.
rename(): Rename columns or index labels.
duplicated(): Check for duplicate rows in a DataFrame.
drop_duplicates(): Remove duplicate rows from a DataFrame.
iloc[]: Select data by integer location.
loc[]: Select data by label.
set_index(): Set a specific column as the index of a DataFrame.
reset_index(): Reset the index of a DataFrame.
rolling(): Calculate rolling statistics for time series data.
shift(): Shift data in a time series.
resample(): Resample time series data.
cumsum(): Calculate the cumulative sum of values.
cumprod(): Calculate the cumulative product of values.
pct_change(): Calculate the percentage change of values.
fillna(): Fill missing values in a DataFrame or Series.
dropna(): Remove missing values from a DataFrame or Series.
pivot_table(): Create a pivot table from data.
melt(): Convert wide data to long format.
unstack(): Unstack a multi-level index.
stack(): Stack data to create a multi-level index.
groupby(): Group data by one or more columns.
apply(): Apply a function to rows or columns of a DataFrame.
mean(): Calculate the mean of values in a DataFrame or Series.
sum(): Calculate the sum of values in a DataFrame or Series.
min(): Calculate the minimum value in a DataFrame or Series.
max(): Calculate the maximum value in a DataFrame or Series.
median(): Calculate the median of values in a DataFrame or Series.
std(): Calculate the standard deviation of values.
var(): Calculate the variance of values in a DataFrame or Series.
quantile(): Calculate quantiles of values in a DataFrame or Series.
cumsum(): Calculate the cumulative sum of values.
cumprod(): Calculate the cumulative product of values.
cummax(): Calculate the cumulative maximum of values.
cummin(): Calculate the cumulative minimum of values.
count(): Count non-null values in a DataFrame or Series.
idxmax(): Return the index of the maximum value.
idxmin(): Return the index of the minimum value.
idxmax(): Return the index of the maximum value.
idxmin(): Return the index of the minimum value.
count(): Count non-null values in a DataFrame or Series.
idxmax(): Return the index of the maximum value.
idxmin(): Return the index of the minimum value.
fillna(): Fill missing values in a DataFrame or Series.
dropna(): Remove missing values from a DataFrame or Series.
pivot_table(): Create a pivot table from data.
melt(): Convert wide data to long format.
unstack(): Unstack a multi-level index.
stack(): Stack data to create a multi-level index.
groupby(): Group data by one or more columns.
apply(): Apply a function to rows or columns of a DataFrame.
mean(): Calculate the mean of values in a DataFrame or Series.
sum(): Calculate the sum of values in a DataFrame or Series.
min(): Calculate the minimum value in a DataFrame or Series.
max(): Calculate the maximum value in a DataFrame or Series.
median(): Calculate the median of values in a DataFrame or Series.
std(): Calculate the standard deviation of values.
var(): Calculate the variance of values in a DataFrame or Series.
quantile(): Calculate quantiles of values in a DataFrame or Series.
cumsum(): Calculate the cumulative sum of values.
cumprod(): Calculate the cumulative product of values.
cummax(): Calculate the cumulative maximum of values.
cummin(): Calculate the cumulative minimum of values.
count(): Count non-null values in a DataFrame or Series.
idxmax(): Return the index of the maximum value.
idxmin(): Return the index of the minimum value.
fillna(): Fill missing values in a DataFrame or Series.
dropna(): Remove missing values from a DataFrame or Series.
pivot_table(): Create a pivot table from data.
melt(): Convert wide data to long format.
unstack(): Unstack a multi-level index.
stack(): Stack data to create a multi-level index.
groupby(): Group data by one or more columns.
apply(): Apply a function to rows or columns of a DataFrame.
mean(): Calculate the mean of values in a DataFrame or Series.
sum(): Calculate the sum of values in a DataFrame or Series.
min(): Calculate the minimum value in a DataFrame or Series.
max(): Calculate the maximum value in a DataFrame or Series.
median(): Calculate the median of values in a DataFrame or Series.
std(): Calculate the standard deviation of values.
var(): Calculate the variance of values in a DataFrame or Series.
quantile(): Calculate quantiles of values in a DataFrame or Series.
cumsum(): Calculate the cumulative sum of values.
cumprod(): Calculate the cumulative product of values.
cummax(): Calculate the cumulative maximum of values.
cummin(): Calculate the cumulative minimum of values.
count(): Count non-null values in a DataFrame or Series.
idxmax(): Return the index of the maximum value.
idxmin(): Return the index of the minimum value.
This list covers a wide range of pandas functions that are commonly used for data manipulation and analysis. Keep in mind that pandas offers many more functions and methods for different tasks, so exploring the official pandas documentation can provide further insights into pandas' capabilities.








# In[ ]:




