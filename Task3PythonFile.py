#!/usr/bin/env python
# coding: utf-8

# 

# In[1]:


pip install umap-learn


# In[1]:





# In[2]:


import re, nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import umap
import plotly.graph_objs as go
import plotly.figure_factory as ff
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objs as go



# In[3]:


# Reading dataset as dataframe
df = pd.read_csv("MovieReviews.csv", encoding='utf-8')
pd.set_option('display.max_colwidth', None) # Setting this so we can see the full content of cells
pd.set_option('display.max_columns', None) # to make sure we can see all the columns in output window
df['sentiment'] = df['sentiment'].map({'positive':1, 'negative':0})


# In[5]:


def cleaner(summary):
    soup = BeautifulSoup(summary, 'lxml') # removing HTML entities such as ‘&amp’,’&quot’,'&gt'; lxml is the html parser and shoulp be installed using 'pip install lxml'
    souped = soup.get_text()
    re1 = re.sub("[^A-Za-z]+"," ", souped) # substituting any non-alphabetic character that repeats one or more times with whitespace

    """
    For more info on regular expressions visit -
    https://docs.python.org/3/howto/regex.html
    """

    tokens = nltk.word_tokenize(re1)
    lower_case = [t.lower() for t in tokens]

    stop_words = set(stopwords.words('english'))
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))

    wordnet_lemmatizer = WordNetLemmatizer()
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    return lemmas


# In[6]:


df['cleaned_review'] = df.review.apply(cleaner)
df = df[df['cleaned_review'].map(len) > 0] # removing rows of length 0 (if any)
print("Printing top 5 rows of dataframe showing original and cleaned reviews....")
print(df[['review','cleaned_review']].head())
df['cleaned_review'] = [" ".join(row) for row in df['cleaned_review'].values] # joining tokens to create strings. TfidfVectorizer does not accept tokens as input
data = df['cleaned_review']
tfidf = TfidfVectorizer(min_df=.0005, ngram_range=(1,3)) # min_df=.0005 means that each ngram (unigram, bigram, & trigram) must be present in at least 30 documents for it to be considered as a token (60000*.0005=30). This is a clever way of feature engineering
tfidf.fit(data) # learn vocabulary of entire data
data_tfidf = tfidf.transform(data) # creating tfidf values
print(tfidf.get_feature_names_out())
print("Shape of tfidf matrix: ", data_tfidf.shape)


# In[7]:


scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_tfidf.toarray())

pca = PCA(n_components=2)
x_pca = pca.fit_transform(data_scaled)




print("Variance explained by each of the n_components: ",pca.explained_variance_ratio_)
print("Total variance explained by the n_components: ",sum(pca.explained_variance_ratio_))
category=list(df['sentiment'])
review = list(df['review'])
data = [go.Scatter(x=x_pca[:,0], y=x_pca[:,1], mode='markers',
                    marker = dict(color=category, colorscale='Rainbow', opacity=0.5),
                                text=[f'Category: {a}' for a in category],
                                hoverinfo='text')]

layout = go.Layout(title = 'PCA Dimensionality Reduction', width = 700, height = 700,
                    xaxis = dict(title='First Principal Component'),
                    yaxis = dict(title='Second Principal Component'))
fig = go.Figure(data=data, layout=layout)
fig.show()


# In[8]:


u = umap.UMAP(n_components=2, n_neighbors=50, min_dist=0.4)
x_umap = u.fit_transform(data_tfidf)

category = list(df['sentiment'])
review = list(df['review'])

data_ = [go.Scatter(x=x_umap[:,0], y=x_umap[:,1], mode='markers',
                    marker = dict(color=df['sentiment'], colorscale='Rainbow', opacity=0.5),
                                text=[f'Category: {a}' for a in category],
                                hoverinfo='text')]

layout = go.Layout(title = 'UMAP Dimensionality Reduction', width = 700, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data_, layout=layout)
fig.show()


# In[7]:




