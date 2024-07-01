#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install umap-learn


# In[2]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objs as go
import plotly.figure_factory as ff
import umap


# In[3]:


dataset = pd.read_csv("online_shoppers_intention.csv")


# In[4]:


dataset.head()


# In[ ]:


print(dataset.head())
print(dataset.shape)
print(dataset.info())
print(dataset.describe())


# In[5]:


X = dataset[['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration']]


# In[6]:


Y = dataset['Revenue'].map({False:0,True:1})


# In[7]:


print(type(X))
print(type(Y))
print(X.shape)
print(Y.shape)


# In[8]:


feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)


# In[9]:


pca = PCA(n_components = 2)
pca.fit(X_scaled)
x_pca = pca.transform(X_scaled)
print("Variance explained by each of the n_components: ",pca.explained_variance_ratio_)
print("Total variance explained by the n_components: ",sum(pca.explained_variance_ratio_))
labels=list(dataset['Revenue'])
data = [go.Scatter(x=x_pca[:, 0], y=x_pca[:, 1], mode='markers',
                    marker=dict(color= Y, colorscale='Rainbow', opacity=0.5),
                    text=[f'label: {rev}' for rev in labels],
                    hoverinfo='text')]

layout = go.Layout(title = 'PCA Dimensionality Reduction', width = 800, height = 800,
                    xaxis = dict(title='First Principal Component'),
                    yaxis = dict(title='Second Principal Component'))
fig = go.Figure(data=data, layout=layout)
fig.show()


# In[10]:


u = umap.UMAP(n_components = 2, n_neighbors=40, min_dist=0.1)
x_umap = u.fit_transform(X_scaled)

data = [go.Scatter(x=x_umap[:,0], y=x_umap[:,1], mode='markers',
                    marker = dict(color=Y, colorscale='Rainbow', opacity=0.5),
                                text=[f'label: {rev}' for rev in labels],
                                hoverinfo='text')]

layout = go.Layout(title = 'UMAP Dimensionality Reduction', width = 800, height = 800,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
fig.show()


# In[ ]:




