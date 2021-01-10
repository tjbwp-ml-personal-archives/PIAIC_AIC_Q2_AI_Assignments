#!/usr/bin/env python
# coding: utf-8

# # Numpy_Assignment_2::

# ## Question:1

# ### Convert a 1D array to a 2D array with 2 rows?

# #### Desired output::

# array([[0, 1, 2, 3, 4],
#         [5, 6, 7, 8, 9]])

# In[7]:


import numpy as np

arr1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
arr2 = arr1.reshape(2,5)
arr2


# ## Question:2

# ###  How to stack two arrays vertically?

# #### Desired Output::
array([[0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1]])
# In[14]:


arr1 = np.array([[0, 1, 2, 3, 4],[5, 6, 7, 8, 9]])
arr2 = np.array([[1, 1, 1, 1, 1],[1, 1, 1, 1, 1]])

np.stack((arr1, arr2), axis=0)


# ## Question:3

# ### How to stack two arrays horizontally?

# #### Desired Output::
array([[0, 1, 2, 3, 4, 1, 1, 1, 1, 1],
       [5, 6, 7, 8, 9, 1, 1, 1, 1, 1]])
# In[16]:


arr1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
arr2 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

np.stack((arr1, arr2), axis=0)


# ## Question:4

# ### How to convert an array of arrays into a flat 1d array?

# #### Desired Output::
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# In[19]:


arr1 = np.array([[0, 1, 2, 3, 4],[5, 6, 7, 8, 9]])
arr1.flatten()


# ## Question:5

# ### How to Convert higher dimension into one dimension?

# #### Desired Output::
array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
# In[22]:


arr1 = np.arange(15).reshape(3,5)
arr1.flatten()


# ## Question:6

# ### Convert one dimension to higher dimension?

# #### Desired Output::
array([[ 0, 1, 2],
[ 3, 4, 5],
[ 6, 7, 8],
[ 9, 10, 11],
[12, 13, 14]])
# In[23]:


arr1 = np.arange(15)
arr1.reshape(5,3)


# ## Question:7

# ### Create 5x5 an array and find the square of an array?

# In[27]:


arr1 = np.arange(25).reshape(5,5)
np.square(arr1)


# ## Question:8

# ### Create 5x6 an array and find the mean?

# In[29]:


arr1 = np.arange(30).reshape(5,6)
print(arr1)
np.mean(arr1)


# ## Question:9

# ### Find the standard deviation of the previous array in Q8?

# In[30]:


arr1 = np.arange(30).reshape(5,6)
print(arr1)
np.std(arr1)


# ## Question:10

# ### Find the median of the previous array in Q8?

# In[31]:


arr1 = np.arange(30).reshape(5,6)
print(arr1)
np.median(arr1)


# ## Question:11

# ### Find the transpose of the previous array in Q8?

# In[32]:


arr1 = np.arange(30).reshape(5,6)
print(arr1)
arr1.transpose()


# ## Question:12

# ### Create a 4x4 an array and find the sum of diagonal elements?

# In[41]:


arr1 = np.arange(16).reshape(4,4)
print(arr1)
arr2 = np.diagonal(arr1)
print (arr2)
arr2.sum()


# ## Question:13

# ### Find the determinant of the previous array in Q12?

# In[43]:


arr1 = np.arange(16).reshape(4,4)
np.linalg.det(arr1)


# ## Question:14

# ### Find the 5th and 95th percentile of an array?

# In[48]:


from numpy import random
arr1 = random.randn(100)
p05 = np.percentile(arr1,5)
p95 = np.percentile(arr1,95)
print (p05, p95)


# ## Question:15

# ### How to find if a given array has any null values?

# In[59]:


arr1 = np.array([1,2,3,4, np.nan])
arr2 = np.sum(arr1)
np.isnan(arr2)


# In[ ]:




