#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install kaggle


# In[2]:


import os
######################## GET IMAGES FROM KAGGLE AND INTO PYTHON ########################
#This function downloads the dataset from Kaggle.com and stores it in the working directory
def DownloadDataset():
    os.environ['KAGGLE_USERNAME'] = 'capstoneddg'
    os.environ['KAGGLE_KEY'] = '43717164337825aa55a1ba0dc8c4b0f2'
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    print("Authenticating API")
    api.authenticate()
    print("Downloading Dataset to Working Directory")
    print("This could take a few minutes.")
    api.dataset_download_file("crowww/a-large-scale-fish-dataset", file_name = '')
    print("Download Complete")


# In[3]:


if os.path.exists(os.path.join(os.getcwd (),'archive.zip')):
    pass
else:
    DownloadDataset()


# In[ ]:




