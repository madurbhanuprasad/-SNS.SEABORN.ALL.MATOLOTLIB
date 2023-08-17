#!/usr/bin/env python
# coding: utf-8

# In[5]:


import plotly.express as px
import pandas as pd
import numpy as np
# Sample data
np.random.seed(42)  # For reproducibility
n = 50
categories = np.random.choice(['A', 'B', 'C'], n)
values = np.random.randint(1, 100, n)

# Create the DataFrame
# Create the DataFrame
data = pd.DataFrame({'Category': categories, 'Value': values})
# Scatter plot
scatter_plot = px.scatter(data, x='Category', y='Value', title='Scatter Plot')
scatter_plot.show()

# Line plot
line_plot = px.line(data, x='Category', y='Value', title='Line Plot')
line_plot.show()

# Bar chart
bar_chart = px.bar(data, x='Category', y='Value', title='Bar Chart')
bar_chart.show()

# Histogram
histogram = px.histogram(data, x='Value', title='Histogram')
histogram.show()

# Box plot
box_plot = px.box(data, x='Category', y='Value', title='Box Plot')
box_plot.show()

# Violin plot
violin_plot = px.violin(data, x='Category', y='Value', title='Violin Plot')
violin_plot.show()

# Pie chart
pie_chart = px.pie(data, names='Category', values='Value', title='Pie Chart')
pie_chart.show()

# Heatmap
heatmap_data = pd.DataFrame({
    'A': [10, 20, 15],
    'B': [25, 30, 35],
    'C': [5, 15, 10]
})
heatmap = px.imshow(heatmap_data, x=heatmap_data.columns, y=heatmap_data.index, title='Heatmap')
heatmap.show()

# 3D Scatter plot
data_3d = pd.DataFrame({
    'X': [1, 2, 3, 4, 5],
    'Y': [10, 20, 15, 25, 30],
    'Z': [5, 15, 10, 20, 25]
})
scatter_3d = px.scatter_3d(data_3d, x='X', y='Y', z='Z', title='3D Scatter Plot')
scatter_3d.show()


# In[6]:


#Here's an example of creating a customized plot using go.Figure from the plotly.graph_objects module:

import plotly.graph_objects as go

# Create a customized plot using go.Figure
fig = go.Figure()

# Add scatter trace to the figure
fig.add_trace(go.Scatter(x=['A', 'B', 'C', 'A', 'B', 'C'],
                         y=[10, 20, 15, 25, 30, 35],
                         mode='markers',
                         marker=dict(size=10),
                         name='Scatter Plot'))

# Customize layout settings
fig.update_layout(title='Customized Scatter Plot',
                  xaxis_title='Category',
                  yaxis_title='Value')

# Show the plot
fig.show()


# In[ ]:


#go.Figure class. We use add_trace() to add a scatter trace to the figure,
and then update_layout() to customize the layout settings such as title
and axis labels. Finally, fig.show() displays the customized plot.

You can further customize the appearance and layout of the plot using 
various attributes and methods provided by the go.Figure class.


# In[7]:


#update_traces() method in Plotly's go.Figure to customize individual trace settings.
import plotly.graph_objects as go

# Create a customized plot using go.Figure
fig = go.Figure()

# Add scatter traces to the figure
fig.add_trace(go.Scatter(x=['A', 'B', 'C', 'A', 'B', 'C'],
                         y=[10, 20, 15, 25, 30, 35],
                         mode='markers',
                         name='Scatter Plot 1'))

fig.add_trace(go.Scatter(x=['A', 'B', 'C', 'A', 'B', 'C'],
                         y=[20, 30, 25, 35, 40, 45],
                         mode='lines',
                         name='Scatter Plot 2'))

# Update individual trace settings
fig.update_traces(marker=dict(size=10),
                  selector=dict(mode='markers'))

fig.update_traces(line=dict(dash='dash'),
                  selector=dict(mode='lines'))

# Customize layout settings
fig.update_layout(title='Customized Scatter Plot',
                  xaxis_title='Category',
                  yaxis_title='Value')

# Show the plot
fig.show()


# In[8]:


import plotly.subplots as sp
import plotly.graph_objects as go

# Create a subplot layout with 2 rows and 2 columns
fig = sp.make_subplots(rows=2, cols=2, subplot_titles=['Subplot 1', 'Subplot 2', 'Subplot 3', 'Subplot 4'])

# Add scatter plots to the subplots
scatter1 = go.Scatter(x=[1, 2, 3], y=[10, 20, 15], mode='markers')
scatter2 = go.Scatter(x=[1, 2, 3], y=[25, 30, 35], mode='markers')
scatter3 = go.Scatter(x=[1, 2, 3], y=[5, 15, 10], mode='markers')
scatter4 = go.Scatter(x=[1, 2, 3], y=[30, 40, 45], mode='markers')

fig.add_trace(scatter1, row=1, col=1)
fig.add_trace(scatter2, row=1, col=2)
fig.add_trace(scatter3, row=2, col=1)
fig.add_trace(scatter4, row=2, col=2)

# Update subplot layout settings
fig.update_layout(title_text='Subplot Layout Example')

# Show the subplot layout
fig.show()


# In[ ]:


#make_subplots to create a 2x2 subplot layout. We then add scatter plots to each
#subplot using the add_trace method, specifying the row and column positions. Finally, we customize the layout
#settings and display the subplot layout using fig.show().


# In[10]:


import plotly.express as px

# Create a DataFrame with animated data
data = pd.DataFrame({
    'Time': [1, 2, 3, 4, 5],
    'X': [10, 15, 20, 25, 30],
    'Y': [5, 10, 15, 20, 25]
})

# Create an animated scatter plot using Plotly Express
fig = px.scatter(data, x='X', y='Y', animation_frame='Time', title='Animated Scatter Plot')

# Show the animated plot
fig.show()


# In[11]:


import plotly.express as px
import pandas as pd

# Create a DataFrame with animated data
data = pd.DataFrame({
    'Time': [1, 2, 3, 4, 5],
    'Value1': [10, 15, 20, 25, 30],
    'Value2': [5, 10, 15, 20, 25]
})

# Create an animated line plot using Plotly Express
fig = px.line(data, x='Time', y=['Value1', 'Value2'], animation_frame='Time', 
              title='Animated Line Plot', labels={'Value1': 'Series 1', 'Value2': 'Series 2'})

# Customize layout and style
fig.update_layout(
    xaxis_title='Time',
    yaxis_title='Value',
    legend_title='Series',
    title_font_size=20
)

# Show the animated plot
fig.show()


# In[12]:


import plotly.express as px

# Create a DataFrame with animated data
data = pd.DataFrame({
    'Time': [1, 2, 3, 4, 5],
    'Value': [10, 15, 20, 25, 30]
})

# Create an animated bar chart using Plotly Express
fig = px.bar(data, x='Time', y='Value', animation_frame='Time', title='Animated Bar Chart')

# Show the animated plot
fig.show()


# In[ ]:


#Choropleth maps are used to display data on a map using different
colors or shading to represent values associated with different geographic 
regions. Plotly Express provides a convenient way to create choropleth maps


# In[14]:


import plotly.express as px

# Sample data with geographical information
data = pd.DataFrame({
    'Country': ['USA', 'Canada', 'Mexico'],
    'Value': [10, 20, 15]
})

# Create a choropleth map
fig = px.choropleth(data_frame=data, locations='Country', locationmode='country names', color='Value',
                    color_continuous_scale='Viridis', title='Choropleth Map')

# Show the choropleth map
fig.show()


# In[16]:


import plotly.express as px

data = pd.DataFrame({
    'Stage': ['Visitors', 'Leads', 'Opportunities', 'Customers'],
    'Count': [1000, 800, 200, 100]
})

fig = px.funnel(data, x='Count', y='Stage')
fig.show()



# In[17]:


from pandas.plotting import scatter_matrix

scatter_matrix(data)


# In[18]:


import plotly.express as px

data = px.data.wind()

fig = px.scatter_polar(data, r="frequency", theta="direction")
fig.show()


# In[ ]:


#Polar plots are useful for visualizing data that has a circular or angular nature.
#Here's how to create a polar plot using Plotly:


# In[19]:


import plotly.figure_factory as ff

data = [
    dict(A=0.1, B=0.2, C=0.7),
    dict(A=0.2, B=0.4, C=0.4),
    dict(A=0.3, B=0.1, C=0.6)
]

fig = ff.create_ternary_contour(data, interpolation='linear', colorscale='Blues')
fig.show()
#Ternary plots are used to display data composed of three parts that sum to a constant. 


# In[20]:


#Waterfall charts are used to visualize cumulative effect of sequentially introduced positive and negative values.

import plotly.graph_objects as go

data = pd.DataFrame({
    'Category': ['Start', 'Income', 'Expenses', 'Net'],
    'Value': [0, 1000, -500, 500]
})

fig = go.Figure(go.Waterfall(
    x=data['Category'],
    y=data['Value']
))

fig.show()


# In[ ]:




