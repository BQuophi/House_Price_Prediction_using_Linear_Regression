#!/usr/bin/env python
# coding: utf-8

# We start by importing the python libraries
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Loading the input data of house prices from a csv file
house_data = pd.read_csv("C:\\Users\\DELL\\Downloads\\house_data.csv")


#  Now, we can check what is in the data we just loaded.
#  We can use the head() function.
#  
# head() returns the first n rows of the dataframe. The default number of elements to display is 5, but you may pass a custom number.
#  
#  Another function is tail() which returns the last n rows.
#  Let's get the first 10 rows

# In[3]:


house_data.head(10)


# In[4]:


# We also have the describe() method which returns description of the data in the DataFrame
house_data.describe()


# In[5]:


# Simple vizualization of the data we have. Any two features can be used just to see how they relate to each other
# We can use scatter plot, bar and pie charts etc.
plt.scatter(house_data['sqft_living'],house_data['price'])


# In[6]:


# Now in the ML program we need to define our input data(x) and output / prediction(y)  that we want to obtain
# Define y first

y = house_data['price']

# In a ML model, we will remove some columns when defining out input data like the price(as it is already in the output or target variable) and some other columns or features. One reason for doing this is because some of these features might not impact the output and are thus irrelevant features.
x =house_data.drop(['price','id','date'], axis=1) # Axis=1 ensures it is columns not rows


# * Training of the data using a Linear Regression. We will use scikit learn/ sklearn library with the linear model to import the LinearRegression class. So we want to create a linear model

# In[7]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()


# Now we need to split the data into training set and testing set.
# 
# We use the train_test_split method. It takes the input-x and output-y
# 
# We use test_size to define the percentage for testing,this can be 20%-0.2, 30%-0.3, etc. Also, random_state is just an initializer, so you can use any value, 2, 1000, etc.

# In[8]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 2)


# In[9]:


# Lets check the shape of our training data, x_train, y_train
print(x_train.shape)# 80% of total data. Has 18 columns out of 21 since we dropped 3 columns
print(y_train.shape) # Has only 1 column, just price
print(x_test.shape) # 20% of total data
print(y_test.shape) # Also 1 column


# In[10]:


# Now the way to train the data is very simple. We call the fit() method on the model and pass the training data to it
model.fit(x_train, y_train) # The model will now go through and learn the data


# In[11]:


# Now that the model has trained on the data, we can test/ score the model against the test data
model.score(x_test, y_test)


# # This score means at 71.56% percent of the time, it is able to predict the house price. This can be improved to a better accuracy
# # For future data we can call model.predict() and get the price of the house of a given input data.

# In[ ]:




