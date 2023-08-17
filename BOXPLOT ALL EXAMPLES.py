#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


data=[0,100,200,500,400,600,10000]
sns.boxplot(data=data)
plt.figure(figsize=(10,7))
plt.xlabel("data")
plt.ylabel("y axis")
plt.title("show")
plt.show()


# In[12]:


#remove outlayers
#convert data in to dataframes

data=pd.DataFrame(data=data,columns=["valves"])
#calculate iqr
q1=data["valves"].quantile(0.25)
q3=data["valves"].quantile(0.75)
iqr=q3-q1
print(iqr)


# In[22]:


#remove outlayers
lowerfence=q1-1.5*iqr
upperfence=q3+1.5*iqr
new_data=data[(data["valves"]>=lowerfence) & (data["valves"]<=upperfence)]
sns.boxplot(data=new_data)


# In[24]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[25]:


# Sample data
data = {
    "Job Category": ["Manager", "Engineer", "Analyst", "Technician", "Manager", "Analyst", "Engineer", "Technician"],
    "Salary": [80000, 70000, 60000, 55000, 85000, 65000, 72000, 56000]
}


# In[27]:


sns.boxplot(x="Salary",y="Job Category",data=data)
plt.show()


# In[28]:


# Create a box plot using Seaborn
plt.figure(figsize=(8, 6))
sns.boxplot(x="Job Category", y="Salary", data=data)

plt.title("Salary Distribution by Job Category")
plt.xlabel("Job Category")
plt.ylabel("Salary")

plt.show()


# In[33]:


# Calculate summary statistics for each job category
summary_stats = df.groupby("Job Category")["Salary"].describe()
print(summary_stats)


# In[32]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data
data = {
    "Job Category": ["Manager", "Engineer", "Analyst", "Technician", "Manager", "Analyst", "Engineer", "Technician"],
    "Salary": [80000, 70000, 60000, 55000, 85000, 65000, 72000, 56000]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Create a box plot using Seaborn
plt.figure(figsize=(8, 6))
sns.boxplot(x="Job Category", y="Salary", data=df)

plt.title("Salary Distribution by Job Category")
plt.xlabel("Job Category")
plt.ylabel("Salary")

plt.show()

# Calculate summary statistics for each job category
summary_stats = df.groupby("Job Category")["Salary"].describe()
print(summary_stats)


# In[34]:


#Assume you have employee data containing information about their ages and the departments 
#they work in. You want to use a box plot to analyze if there are significant differences 
#in the ages of employees among different departments.


# In[35]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[36]:


data = {
    "Department": ["HR", "Engineering", "Marketing", "Engineering", "HR", "Marketing", "Engineering"],
    "Age": [32, 28, 35, 42, 29, 37, 31]
}


# In[37]:


#create a dataframe
data=pd.DataFrame(data)


# In[38]:


#create a boxplot using seaborn
sns.boxplot(x="Department",y="Age",data=data)


# In[42]:


sns.boxplot(x="Department",y="Age",data=data)
plt.title("Age Distribution by Department")
plt.xlabel("Department")
plt.ylabel("Age")
plt.show()


# In[43]:


#you have a dataset containing information about house prices and the neighborhoods
#they are located in. You want to use a box plot to identify potential outliers 
#in the distribution of house prices for different neighborhoods.


# In[44]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample house price data with real neighborhood names
data = {
    "Neighborhood": ["Downtown", "Suburb", "Downtown", "Suburb", "Urban", "Suburb", "Downtown", "Urban", "Urban"],
    "Price": [300000, 350000, 320000, 280000, 800000, 330000, 310000, 360000, 310000]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Create a box plot using Seaborn
plt.figure(figsize=(8, 6))
sns.boxplot(x="Neighborhood", y="Price", data=df)

plt.title("House Price Distribution by Neighborhood")
plt.xlabel("Neighborhood")
plt.ylabel("Price ($)")

plt.show()


# In[45]:


#Assume you have a dataset with information about monthly sales revenue and the
#corresponding regions. You want to use a box plot to analyze how the distribution
#of monthly sales revenue varies between different regions.


# In[56]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample sales revenue data by region and month
data = {
    "Region": ["North", "South", "North", "South", "North", "South"],
    "Month": ["January", "January", "February", "February", "March", "March"],
    "Revenue": [120000, 90000, 110000, 85000, 115000, 95000]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Create a box plot using Seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(x="Month", y="Revenue", hue="Region",data=df)

plt.title("Monthly Sales Revenue by Region")
plt.xlabel("Month")
plt.ylabel("Revenue ($)")

plt.legend(title="Region")

plt.show()


# In[57]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample crime rate data by neighborhood type
data = {
    "Neighborhood Type": ["Urban", "Suburb", "Urban", "Suburb", "Rural", "Urban"],
    "Crime Rate": [25, 10, 20, 8, 5, 15]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Create a box plot using Seaborn
plt.figure(figsize=(8, 6))
sns.boxplot(x="Neighborhood Type", y="Crime Rate", data=df)

plt.title("Crime Rate Distribution by Neighborhood Type")
plt.xlabel("Neighborhood Type")
plt.ylabel("Crime Rate")

plt.show()


# In[59]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample time series dataset with stock prices
data = {
    "Date": ["2022-01-01", "2022-01-02", "2022-01-03", "2022-12-29", "2022-12-30", "2022-12-31"],
    "Stock Price": [120, 125, 118,  135, 130, 138]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Convert the 'Date' column to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Extract month from the 'Date' column
df["Month"] = df["Date"].dt.month

# Create a box plot for each month's stock prices
plt.figure(figsize=(10, 6))
sns.boxplot(x="Month", y="Stock Price", data=df)

# Highlight extreme values using the 'fliersize' parameter
sns.swarmplot(x="Month", y="Stock Price", data=df, color="red", size=6)

plt.title("Box Plot of Monthly Stock Prices with Outliers Highlighted")
plt.xlabel("Month")
plt.ylabel("Stock Price")

plt.show()


# In[ ]:





# In[65]:


#2. Multivariate Outlier Analysis:
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample multivariate dataset with numerical variables
data = {
    "Variable1": [5, 8, 6, 12, 9, 7, 10, 11, 15, 8],
    "Variable2": [20, 25, 18, 30, 22, 23, 26, 28, 35, 24],
    "Variable3": [300, 250, 280, 320, 290, 310, 330, 315, 280, 300]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Create a pairplot with scatter plots of all combinations of variables
sns.set(style="ticks")
sns.pairplot(df)

plt.suptitle("Pairplot for Multivariate Outlier Analysis", y=1.02)

plt.show()


# In[66]:


#The pairplot can help you identify potential relationships and patterns among variables, 
#including outliers, which can guide further exploration and analysis.


# In[ ]:





# In[67]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Sample dataset with a skewed variable
data = {
    "Skewed_Variable": np.random.exponential(scale=3, size=1000)
}

# Create a DataFrame
df = pd.DataFrame(data)

# Create a distplot to visualize the skewed distribution
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")
sns.distplot(df["Skewed_Variable"], kde=False, hist_kws={"alpha": 0.7})

plt.title("Original Skewed Distribution")
plt.xlabel("Skewed Variable")
plt.ylabel("Frequency")

plt.show()

# Apply a log transformation to mitigate skewness
df["Transformed_Variable"] = np.log(df["Skewed_Variable"])

# Create box plots to compare the original and transformed distributions
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")
sns.boxplot(data=df[["Skewed_Variable", "Transformed_Variable"]])

plt.title("Comparison of Original and Transformed Distributions")
plt.ylabel("Value")

plt.xticks(ticks=[0, 1], labels=["Original", "Transformed"])

plt.show()


# In[68]:


#By setting kde=False in the sns.distplot function, you'll plot just the histogram 
#without the kernel density estimate, which can be more appropriate when dealing with highly skewed data.


# In[ ]:


#dataset with daily website traffic over a year, and you want to 
#identify anomalies or outliers in daily traffic. Create a time series
#plot using Seaborn and highlight days with unusually high or low traffic compared to the seasonal pattern.


# In[69]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Simulated dataset with daily website traffic over a year
dates = pd.date_range(start="2022-01-01", end="2022-12-31", freq="D")
traffic = np.sin(np.linspace(0, 2*np.pi, num=365)) + np.random.normal(0, 0.2, size=365)

# Create a DataFrame
df = pd.DataFrame({"Date": dates, "Traffic": traffic})

# Create a time series plot using Seaborn
plt.figure(figsize=(12, 6))
sns.lineplot(x="Date", y="Traffic", data=df)
plt.title("Daily Website Traffic Over a Year")
plt.xlabel("Date")
plt.ylabel("Traffic")

# Detect anomalies by comparing traffic to a rolling mean
rolling_mean = df["Traffic"].rolling(window=30, min_periods=1, center=True).mean()
anomalies = df[abs(df["Traffic"] - rolling_mean) > 1.5 * rolling_mean.std()]

# Highlight anomalies with red points
sns.scatterplot(x="Date", y="Traffic", data=anomalies, color="red", marker="x", s=50, label="Anomalies")

plt.legend()
plt.show()


# In[ ]:


# this example, we're using Seaborn to create a time series plot of daily website traffic over a year. We're simulating the data with a seasonal pattern using sine function and adding some random noise. Anomalies are detected by comparing the daily traffic to a rolling mean and marking points with traffic values significantly different from the mean as anomalies.

The sns.scatterplot function is then used to plot the detected anomalies in 
red with an "x" marker. This allows you to visually identify days with unusually 
high or low traffic compared to the seasonal pattern.

Keep in mind that anomaly detection can be more complex depending on the 
characteristics of your data and the specific algorithm or statistical approach you use. 
This example demonstrates a basic approach to identifying anomalies based on the difference from a rolling mean.



#df["Traffic"]: This part accesses the "Traffic" column of the DataFrame df, which contains the daily website traffic values.

.rolling(window=30, min_periods=1, center=True): This part creates a rolling window of size 30 days.
    The window parameter determines the size of the window, and min_periods specifies the minimum number 
    of non-null data points required in the window. The center parameter is set to True, which means that
    the rolling window's center will be aligned with the time step it corresponds to.

.mean(): This computes the mean value of the data points within the rolling window for each time step.

Putting it all together, the code calculates the rolling mean of the daily website traffic over a window of 30 days.
The rolling mean is computed for each time step (day) in the dataset. This rolling mean can help you identify the
general trend and seasonal pattern in the data, which can be used for anomaly detection by comparing individual data 
points to this rolling mean.

In the context of the previous example, the code segment is used to calculate the rolling mean of daily website traffic,
which is then used to detect anomalies by comparing each day's traffic to this rolling mean.



#f["Traffic"]: This part accesses the "Traffic" column of the DataFrame df, which contains the daily website traffic values.

rolling_mean: This refers to the calculated rolling mean of the daily website traffic values over a window of 30 days,
    as computed using the previous line of code.

abs(df["Traffic"] - rolling_mean): This calculates the absolute difference between the original daily traffic values 
    and the corresponding rolling mean values. It quantifies how much each day's traffic deviates from the rolling mean.

rolling_mean.std(): This calculates the standard deviation of the rolling mean. The standard deviation is a measure
    of the variability of the rolling mean values.

1.5 * rolling_mean.std(): This calculates a threshold value as 1.5 times the standard deviation of the rolling mean.
    This threshold is used to identify anomalies that deviate significantly from the rolling mean.

abs(df["Traffic"] - rolling_mean) > 1.5 * rolling_mean.std(): This condition checks whether the absolute difference
    between the original traffic values and the rolling mean is greater than the threshold. If this condition is True
    for a specific day, it means that the day's traffic is significantly different from the rolling mean and might be
    an anomaly.

df[...]: This part selects rows from the DataFrame df where the condition specified in the square brackets is True.
    In other words, it filters the dataset to include only those rows (days) where the traffic values are anomalies 
    based on the specified condition.

In summary, the code segment identifies and selects anomalies in the daily website traffic data by comparing each
day's traffic to the rolling mean and using a threshold of 1.5 times the standard deviation of the rolling mean. 
Days with traffic values that deviate significantly from the rolling mean are considered anomalies and are stored
in the anomalies DataFrame for further visualization or analysis.











# In[ ]:


#ou have a dataset with a variable that contains outliers, and you want to explore different
#strategies for handling them. Use Seaborn to create box plots for the original and transformed data 
#(e.g., log-transformed) to compare the impact of outlier treatment on the distribution.


# In[70]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Simulated dataset with a variable containing outliers
data = {
    "Variable": np.concatenate([np.random.normal(50, 10, size=100),
                                np.random.normal(150, 20, size=10)])  # Adding outliers
}

# Create a DataFrame
df = pd.DataFrame(data)

# Create box plots for the original and transformed (log-transformed) data
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, y="Variable")
plt.title("Box Plot of Original Data")
plt.ylabel("Value")

plt.figure(figsize=(10, 6))
df["Log Transformed"] = np.log(df["Variable"])
sns.boxplot(data=df, y="Log Transformed")
plt.title("Box Plot of Log-Transformed Data")
plt.ylabel("Value (Log Transformed)")

plt.tight_layout()
plt.show()


# In[ ]:


#Hierarchical Outlier Analysis:

Assume you have a dataset with sales data for different product categories
across multiple regions. Use Seaborn's catplot to create categorical scatter
plots for each region's sales in each product category, and identify any regions 
or categories with outlier sales patterns.


# In[ ]:





# In[71]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Simulated dataset with sales data for different product categories across regions
data = {
    "Region": np.repeat(["North", "South", "East", "West"], 50),
    "Product Category": np.tile(["Category A", "Category B", "Category C", "Category D"], 50),
    "Sales": np.random.randint(1000, 10000, size=200)
}

# Create a DataFrame
df = pd.DataFrame(data)

# Create categorical scatter plots using catplot
plt.figure(figsize=(10, 6))
sns.catplot(data=df, x="Product Category", y="Sales", hue="Region", kind="swarm", dodge=True)
plt.title("Categorical Scatter Plots of Sales by Product Category and Region")
plt.ylabel("Sales")

plt.show()


# In[ ]:


#In this example, we're simulating a dataset with sales data for different product categories across four regions. We're using sns.catplot to create categorical scatter plots to visualize the sales distribution for each region within each product category. The kind="swarm" parameter specifies that we want to create a swarm plot, which positions individual data points while avoiding overlap.

Hierarchical outlier analysis involves visually examining these scatter plots to identify any 
regions or categories with outlier sales patterns. Outliers may appear as data points that 
are significantly different from the overall distribution within a specific category and region.
By plotting sales data in this hierarchical manner, you can identify potential anomalies in sales
patterns across regions and categories.






sns.catplot(data=df, x="Product Category", y="Sales", hue="Region", kind="swarm", dodge=True)
data=df: This specifies the DataFrame df that contains the data you want to visualize.

x="Product Category": This specifies the variable you want to represent on the x-axis, which is the "Product Category" in this case.

y="Sales": This specifies the variable you want to represent on the y-axis, which is the "Sales" in this case.

hue="Region": This parameter adds color differentiation to the plot based on the "Region" variable. Each region's data points will have a different color, making it easier to distinguish regions.

kind="swarm": This specifies the type of plot you want to create. "Swarm" plot positions each data point
    along the categorical axis to avoid overlap, providing a clear representation of the distribution of 
    data points within each category.

dodge=True: This parameter ensures that data points are slightly separated along the x-axis 
    when the hue variable is used (in this case, "Region"). It helps prevent overlapping data
    points when there are multiple categories within a hue category.

Putting it all together, the code generates a categorical scatter plot using Seaborn's catplot. It visualizes the sales data across different product categories for each region. Each data point represents a specific sales value, and the swarm plot positions them along the categorical axis while avoiding overlap. Different colors indicate different regions, making it easy to compare sales distributions across categories and regions.

This type of visualization is useful for identifying patterns, trends, and potential outliers within categorical data, allowing you to gain insights into relationships between different variables.






