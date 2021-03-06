
# coding: utf-8

# In[1]:

import numpy as np
import scipy.ndimage

import json
import matplotlib.pyplot as plt
from tqdm import tqdm


# In[2]:

data = {}
with open('Documents/new_val_set.json') as f:
    data = json.load(f)


# In[3]:

def group_list(l, group_size):
    """
    :param l:           list
    :param group_size:  size of each group
    :return:            Yields successive group-sized lists from l.
    """
    for i in range(0, len(l), group_size):
        yield l[i:i+group_size]


# In[4]:

batches = group_list(list(data.keys()), 10)
ims = np.zeros((10, 224, 224, 3))
num = 0
for batch in tqdm(batches):
    num += 1
    for path, ind in zip(batch, range(10)):
        im = scipy.ndimage.imread(path)
        im = im[:1944, :1944, :]
        ims[ind] = scipy.misc.imresize(im, (224,224))
    np.savez('/home/data2/vision6/azlu/val_data_10/img_val_batch_{0}.npz'.format(num), images=ims)


# In[ ]:



