#!/usr/bin/env python
# coding: utf-8

# In[8]:


pip install kaggle


# In[9]:


import os
from kaggle.api.kaggle_api_extended import KaggleApi
def DownloadDataset():
    
    api = KaggleApi()
    print("Authenticating API")
    api.authenticate()
    print("Downloading Dataset to Working Directory")
    print("This could take a few minutes.")
    api.dataset_download_file("crowww/a-large-scale-fish-dataset", file_name = '')
    print("Download Complete")


# In[10]:


if os.path.exists(os.path.join(os.getcwd (),'archive.zip')):
    pass
else:
    DownloadDataset()


# In[ ]:




