#!/usr/bin/env python
# coding: utf-8

# # **Assignment For Numpy**

# Difficulty Level **Beginner**

# 1. Import the numpy package under the name np

# In[13]:


import numpy as np


# 2. Create a null vector of size 10 

# In[10]:


v = np.zeros(10)


# 3. Create a vector with values ranging from 10 to 49

# In[11]:


v = np.arange(10,50)


# 4. Find the shape of previous array in question 3

# In[17]:


v.shape


# 5. Print the type of the previous array in question 3

# In[23]:


v.dtype


# 6. Print the numpy version and the configuration
# 

# In[28]:


print(np.__version__)
print(np.show_config())


# 7. Print the dimension of the array in question 3
# 

# In[29]:


print (v.ndim)


# 8. Create a boolean array with all the True values

# In[32]:


x = np.full((10,10), True, dtype=bool)
x


# 9. Create a two dimensional array
# 
# 
# 

# In[35]:


xy = np.empty((10,10))


# 10. Create a three dimensional array
# 
# 

# In[37]:


xyz = np.empty((2,3,4))


# Difficulty Level **Easy**

# 11. Reverse a vector (first element becomes last)

# In[40]:


v = np.array([2,4,7,9,99])
reverse_v = v[::-1]


# 12. Create a null vector of size 10 but the fifth value which is 1 

# In[6]:


vx = np.zeros(10)
vx[4]=1
vx


# 13. Create a 3x3 identity matrix

# In[12]:


m2d = np.identity(3)
m2d


# 14. arr = np.array([1, 2, 3, 4, 5]) 
# 
# ---
# 
#  Convert the data type of the given array from int to float 

# In[18]:


arr = np.array([1, 2, 3, 4, 5])
arr1 = arr.astype('float64')
arr1


# 15. arr1 =          np.array([[1., 2., 3.],
# 
#                     [4., 5., 6.]])  
#                       
#     arr2 = np.array([[0., 4., 1.],
#      
#                    [7., 2., 12.]])
# 
# ---
# 
# 
# Multiply arr1 with arr2
# 

# In[20]:


arr1 = np.array([[1., 2., 3.],

                [4., 5., 6.]])  

arr2 = np.array([[0., 4., 1.],

               [7., 2., 12.]])


arr1 * arr2


# 16. arr1 = np.array([[1., 2., 3.],
#                     [4., 5., 6.]]) 
#                     
#     arr2 = np.array([[0., 4., 1.], 
#                     [7., 2., 12.]])
# 
# 
# ---
# 
# Make an array by comparing both the arrays provided above

# In[21]:


arr1 = np.array([[1., 2., 3.],

                [4., 5., 6.]]) 

arr2 = np.array([[0., 4., 1.],

                [7., 2., 12.]])

arr1 == arr2


# 17. Extract all odd numbers from arr with values(0-9)

# In[27]:


array1 = np.arange(10)
array1 [array1 % 2 == 1]


# 18. Replace all odd numbers to -1 from previous array

# In[28]:


array1 [array1 % 2 == 1]=-1
array1


# 19. arr = np.arange(10)
# 
# 
# ---
# 
# Replace the values of indexes 5,6,7 and 8 to **12**

# In[36]:


array1 [4:8]=12
array1


# 20. Create a 2d array with 1 on the border and 0 inside

# In[81]:


array_2d = np.ones([5,5])
array_2d[1:-1,1:-1]=0
array_2d


# Difficulty Level **Medium**

# 21. arr2d = np.array([[1, 2, 3],
# 
#                     [4, 5, 6], 
# 
#                     [7, 8, 9]])
# 
# ---
# 
# Replace the value 5 to 12

# In[82]:


rr2d = np.array([[1, 2, 3],

            [4, 5, 6], 

            [7, 8, 9]])
rr2d[1:-1,1:-1]=12
rr2d


# 22. arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# 
# ---
# Convert all the values of 1st array to 64
# 

# In[95]:


arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d[:-1,:-1][:]=64
arr3d


# 23. Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it

# In[130]:


arr_2D = np.arange(9).reshape(3,3)
sliced_array1 = arr_2D[0, :]
arr_2D
sliced_array1


# 24. Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it

# In[136]:


arr_2D = np.arange(9).reshape(3,3)
sliced_array2 = arr_2D[1,1:-1]
sliced_array2


# 25. Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows

# In[154]:


arr_2D = np.arange(9).reshape(3,3)

sliced_array3 = arr_2D[:2,2:]
sliced_array3


# 26. Create a 10x10 array with random values and find the minimum and maximum values

# In[163]:


from numpy import random

arr_rnd = random.randint(100, size=(10, 10))
print(arr_rnd)
mi = np.min(arr_rnd)
ma = np.max(arr_rnd)
print(mi, ma)


# 27. a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])
# ---
# Find the common items between a and b
# 

# In[165]:


a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
np.intersect1d(a,b)


# 28. a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])
# 
# ---
# Find the positions where elements of a and b match
# 
# 

# In[170]:


a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
np.where(a==b)


# 29.  names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])  data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will**
# 

# In[ ]:





# 30. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will** and **Joe**
# 
# 

# In[ ]:





# Difficulty Level **Hard**

# 31. Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.

# In[15]:


array_2D = np.arange(1.,16.).reshape(5,3)
array_2D


# 32. Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.

# In[21]:


array_3D = np.arange(1.,17.).reshape(2,2,4)
array_3D


# 33. Swap axes of the array you created in Question 32

# In[22]:


array_3D = np.arange(1.,17.).reshape(2,2,4)
np.swapaxes(array_3D,0,2)


# 34. Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0

# In[26]:


array = np.arange(1,10)
arrsq = np.sqrt(array)
arrsq[arrsq<0.5]=0.
arrsq


# 35. Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays

# In[35]:


from numpy import random

arr1 = random.randn(12)
arr2 = random.randn(12)
array = np.fmax(arr1,arr2)
print (arr1)
print (arr2)
print (array)


# 36. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# 
# ---
# Find the unique names and sort them out!
# 

# In[36]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
names_unique = np.unique(names)
print (names_unique)


# 37. a = np.array([1,2,3,4,5])
# b = np.array([5,6,7,8,9])
# 
# ---
# From array a remove all items present in array b
# 
# 

# In[48]:


a = np.array([1,2,3,4,5])
b = np.array([5,6,7,8,9])

a = [i for i in a if i not in b]
print (a)


# 38.  Following is the input NumPy array delete column two and insert following new column in its place.
# 
# sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]]) 
# 
# newColumn = numpy.array([[10,10,10]])
#     

# In[97]:


sample = np.array([[34,43,73],[82,22,12],[53,94,66]])
i=1
sample[:1][:1]


# 39. x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])
# 
# 
# ---
# Find the dot product of the above two matrix
# 

# In[70]:


x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
np.dot(x,y)


# 40. Generate a matrix of 20 random values and find its cumulative sum

# In[67]:


matrix = random.randn(20)
sum=np.sum(matrix)
print (matrix)
print (sum)


# In[ ]:




