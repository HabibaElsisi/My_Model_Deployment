#!/usr/bin/env python
# coding: utf-8

# # **Content-Based Recommendation System Notebook**
# - In this notebook, we will explore and implement a content-based recommendation system. Content-based recommendation systems suggest items to users based on the characteristics of the items and a profile of the user's preferences. 
# - This approach is particularly useful when we have a lot of information about the items and the users' preferences. We will build a simple content-based recommendation system using Python and the scikit-learn library.

# **Table of Contents**
# 1. Introduction
#     - What is a Content-Based Recommendation System?
#     - How Does it Work?
#     - Data Preparation
# 
# 
# 2. Dataset
#     - Feature Extraction
#     - Data Preprocessing
#     - Building the Content-Based Recommendation System
# 
# 
# 3. TF-IDF 
#     - Vectorization
#     - Cosine Similarity
#     - Recommending Items
# 
# 
# 4. Evaluation
#     - Evaluation Metrics
# 
# 
# 5. Conclusion
#     - Summary
#  --------------------------------------------

# ## **1. Introduction**
# - **What is a Content-Based Recommendation System?**
#     - A content-based recommendation system recommends items to users based on the content or characteristics of the items. This type of recommendation system focuses on understanding the properties of items and learning user preferences from the items they have interacted with in the past.
# 
# 
# - **How Does it Work?**
#     - The working principle of a content-based recommendation system can be summarized in a few steps:
#         1. **Feature Extraction**: Extract relevant features from the items. For example, in a book recommendation system, features could include title, author, and category
# 
#         2. **User Profile**: Create a user profile based on their interactions with items. This profile is essentially a summary of the features of items the user has liked or interacted with in the past.
# 
#         3. **Recommendation**: Calculate the similarity between the user profile and each item's features. Items that are most similar to the user profile are recommended.

# ## 2. **Data Preparation**
# **Dataset**
#    - We will use a dataset containing books information, including titles, authors, and categories.

# In[1]:


# Import needed modules
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


# Read data
df = pd.read_csv("E:/data.csv")


# - Let's optain some analysis

# In[3]:


# printing the first 5 rows of the dataframe
df.head()


# In[4]:


# Get data information
df.info()


# **Feature Extraction**
# - We will extract relevant features from the dataset, such as book titles and authors.

# In[5]:


# Selecting the relevant features for recommendation
selected_features = ['title','authors','categories','published_year']
print(selected_features)


# **Data Preprocessing**
# - Before building the recommendation system, we need to preprocess the data. This may include text cleaning, handling missing values, and tokenization.

# In[6]:


# Replacing the null valuess with null string
for feature in selected_features:
    df[feature] = df[feature].fillna('')


# In[7]:


# combining all the 4 selected features
combined_features = df['title'] + ' ' + df['categories'] + ' ' + df['authors'] + ' ' + f"{df['published_year']}"
combined_features


# ## 3. **Building the Content-Based Recommendation System**
# **TF-IDF Vectorization**
# - We use TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to convert text features (descriptions) into numerical vectors. 
# - TF-IDF gives more weight to terms that are important in a specific document and less weight to common terms.
# 

# In[8]:


# converting the text data to feature vectors
vectorizer = TfidfVectorizer()

feature_vectors = vectorizer.fit_transform(combined_features)


# In[9]:


print(feature_vectors)


# **Cosine Similarity**
# - We compute the cosine similarity between the TF-IDF vectors of items. Cosine similarity measures the cosine of the angle between two non-zero vectors and is used to determine how similar two items are based on their feature vectors.

# In[10]:


# getting the similarity scores using cosine similarity
similarity = cosine_similarity(feature_vectors, feature_vectors)


# In[11]:


print(similarity)


# **Test your Recommendation System**

# In[12]:


# creating a list with all the book names given in the dataset

list_of_all_titles = df['title'].tolist()
print(list_of_all_titles)


# In[13]:


# getting the book name from the user
book_name = input(' Enter your favourite book name : ') # input: Rage of angels


# In[14]:


# finding the close match for the book name given by the user
find_close_match = difflib.get_close_matches(book_name, list_of_all_titles)
print(find_close_match)


# In[15]:


# finding the index of the book with title
close_match = find_close_match[0]
index_of_the_book = df[df.title == close_match].index[0]


# In[16]:


# getting a list of similar books
similarity_score = list(enumerate(similarity[index_of_the_book]))
print(similarity_score)


# In[17]:


# sorting the books based on their similarity score
sorted_similar_books = sorted(similarity_score, key = lambda x:x[1], reverse = True) 
print(sorted_similar_books)


# In[18]:


top_sim = sorted_similar_books[:5]
top_sim


# In[19]:


# print the name of similar books based on the index
i = 1

for book in sorted_similar_books:
    index = book[0]
    title_from_index = df[df.index==index]['title'].values[0]
    if (i < 6):
        print(i, '-', title_from_index)
        i += 1


# ## **Full Recommendation System**

# In[20]:


book_name = input(' Enter your favourite book name : ')

list_of_all_titles = df['title'].tolist()

find_close_match = difflib.get_close_matches(book_name, list_of_all_titles)

close_match = find_close_match[0]

index_of_the_book = df[df.title == close_match].index[0]

similarity_score = list(enumerate(similarity[index_of_the_book]))

sorted_similar_books = sorted(similarity_score, key = lambda x:x[1], reverse = True) 

print('Books suggested for you : \n')

i = 1

for book in sorted_similar_books:
    index = book[0]
    title_from_index = df[df.index==index]['title'].values[0]
    if (i < 30):
        print(i, '.',title_from_index)
        i+=1


# In[ ]:




