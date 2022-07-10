#!/usr/bin/env python
# coding: utf-8

# In[ ]:


https://www.kaggle.com/code/arashnic/simple-lstm-regression


# In[16]:


import time
import numpy as np
import pandas as pd
import pandas_datareader as pdr

from keras.layers import LSTM
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, Activation, Dropout

from sklearn.preprocessing import MinMaxScaler


# In[ ]:


https://www.kaggle.com/code/arashnic/simple-lstm-regression


# In[17]:


get_ipython().system('pip install TimeDistributed')


# In[13]:


# fails but should work
from tensorflow.keras import datasets


# In[14]:


# succeeds
import tensorflow as tf
tf.keras.datasets


# In[15]:


# succeeds
from tensorflow.python.keras import datasetst


# In[ ]:




