#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#
Certainly! Scatter plots are used to visualize the relationship between two
numerical variables. Here are some examples of scatter plots using Python and Seaborn:


# In[1]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create sample data
x = [10, 20, 30, 40, 50]
y = [25, 30, 40, 50, 60]

# Create a scatter plot
plt.figure(figsize=(6, 4))
sns.scatterplot(x=x, y=y)
plt.title("Basic Scatter Plot")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# In[2]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create sample data
x = [10, 20, 30, 40, 50]
y = [25, 30, 40, 50, 60]

# Create a scatter plot with regression line
plt.figure(figsize=(6, 4))
sns.regplot(x=x, y=y)
plt.title("Scatter Plot with Regression Line")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# In[3]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create sample data with categorical variable
data = {
    "X": [10, 20, 30, 40, 50],
    "Y": [25, 30, 40, 50, 60],
    "Category": ["A", "A", "B", "B", "A"]
}

# Create a scatter plot with hue for categorical variable
plt.figure(figsize=(6, 4))
sns.scatterplot(data=data, x="X", y="Y", hue="Category")
plt.title("Scatter Plot with Hue")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(title="Category")
plt.show()


# In[4]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create sample data with size variable
data = {
    "X": [10, 20, 30, 40, 50],
    "Y": [25, 30, 40, 50, 60],
    "Size": [100, 200, 300, 400, 500]
}

# Create a scatter plot with size for another numerical variable
plt.figure(figsize=(6, 4))
sns.scatterplot(data=data, x="X", y="Y", size="Size", sizes=(50, 300))
plt.title("Scatter Plot with Size")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# In[5]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Create linearly related data
np.random.seed(42)
x = np.linspace(0, 10, 50)
y = 2 * x + 3 + np.random.normal(0, 1, 50)

# Create a scatter plot with regression line
plt.figure(figsize=(6, 4))
sns.regplot(x=x, y=y, scatter_kws={"color": "blue"}, line_kws={"color": "red"})
plt.title("Scatter Plot with Linear Relationship and Regression Line")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# In[6]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Create negative linearly related data with categories
data = {
    "X": [10, 20, 30, 40, 50],
    "Y": [50, 40, 30, 20, 10],
    "Category": ["A", "A", "B", "B", "A"]
}

df = pd.DataFrame(data)

# Create a scatter plot with hue for categorical variable
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x="X", y="Y", hue="Category")
plt.title("Scatter Plot with Negative Linear Relationship and Hue")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(title="Category")
plt.show()


# In[7]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Create data with non-linear relationship
np.random.seed(42)
x = np.linspace(0, 10, 50)
y = 3 * np.sin(x) + np.random.normal(0, 0.5, 50)
size = np.random.randint(20, 100, 50)
style = np.where(x > 5, "A", "B")

# Create a scatter plot with size and style
plt.figure(figsize=(6, 4))
sns.scatterplot(x=x, y=y, size=size, style=style, palette=["blue", "orange"])
plt.title("Scatter Plot with Non-Linear Relationship, Size, and Style")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(title="Style", loc="upper left")
plt.show()


# In[8]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Create random data for three variables
np.random.seed(42)
x = np.random.rand(50)
y1 = 2 * x + np.random.normal(0, 0.1, 50)
y2 = -3 * x + np.random.normal(0, 0.1, 50)
y3 = np.random.normal(0, 0.1, 50)

# Create subplots with multiple scatter plots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
sns.scatterplot(x=x, y=y1, ax=axes[0])
sns.scatterplot(x=x, y=y2, ax=axes[1])
sns.scatterplot(x=x, y=y3, ax=axes[2])
axes[0].set_title("Scatter Plot 1")
axes[1].set_title("Scatter Plot 2")
axes[2].set_title("Scatter Plot 3")
plt.tight_layout()
plt.show()


# In[9]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Generate 3D data
np.random.seed(42)
n_points = 100
x = np.random.rand(n_points)
y = np.random.rand(n_points)
z = 3 * x + 2 * y + np.random.normal(0, 0.1, n_points)

# Create a 3D scatter plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c=z, cmap='viridis', s=50)
ax.set_title("3D Scatter Plot")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()


# In[11]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load Iris dataset
iris = sns.load_dataset("iris")

# Create a pairplot with scatter plots and histograms
sns.pairplot(iris, hue="species", diag_kind="kde", markers=["o", "s", "D"])
plt.suptitle("Pairplot with Scatter Plots and Histograms", y=1.02)
plt.show()


# In[12]:


import seaborn as sns
import matplotlib.pyplot as plt

# Load tips dataset
tips = sns.load_dataset("tips")

# Create a joint distribution plot using hexbin
sns.jointplot(data=tips, x="total_bill", y="tip", kind="hex", gridsize=20)
plt.title("Joint Distribution with Hexbin Plot")
plt.show()


# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt

# Load tips dataset
tips = sns.load_dataset("tips")

# Create a facet grid of scatter plots
g = sns.FacetGrid(tips, col="time", hue="day", col_wrap=2)
g.map(sns.scatterplot, "total_bill", "tip", alpha=0.7)
g.add_legend(title="Day")
plt.suptitle("Facet Grid of Scatter Plots by Time", y=1.02)
plt.show()


# In[14]:


import seaborn as sns
import matplotlib.pyplot as plt

# Load Iris dataset
iris = sns.load_dataset("iris")

# Create a scatter matrix
sns.set(style="ticks")
sns.pairplot(iris, hue="species")
plt.suptitle("Scatter Matrix", y=1.02)
plt.show()


# In[15]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Create sample data
data = {
    "X": [10, 20, 30, 40, 50],
    "Y": [25, 30, 40, 50, 60],
    "Size": [100, 200, 300, 400, 500],
    "Color": ["Red", "Blue", "Green", "Yellow", "Orange"]
}

df = pd.DataFrame(data)

# Create a bubble chart with size and color mapping
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="X", y="Y", size="Size", hue="Color")
plt.title("Bubble Chart with Size and Color Mapping")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(title="Color")
plt.show()


# In[16]:


import seaborn as sns
import matplotlib.pyplot as plt

# Load Tips dataset
tips = sns.load_dataset("tips")

# Create a grouped scatter plot with linear regression
plt.figure(figsize=(8, 6))
sns.regplot(data=tips, x="total_bill", y="tip", scatter_kws={"s": 50})
plt.title("Grouped Scatter Plot with Linear Regression")
plt.xlabel("Total Bill")
plt.ylabel("Tip")
plt.show()


# In[ ]:




