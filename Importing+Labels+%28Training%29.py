
# coding: utf-8

# In[1]:

import numpy as np
import scipy.ndimage
import scipy.misc
import json
import matplotlib.pyplot as plt
from tqdm import tqdm


# In[2]:

data = {}
with open('Documents/new_train_set.json') as f:
    data = json.load(f)


# In[3]:

print(data.values())


# In[4]:

def group_list(l, group_size):
    """
    :param l:           list
    :param group_size:  size of each group
    :return:            Yields successive group-sized lists from l.
    """
    for i in range(0, len(l), group_size):
        yield l[i:i+group_size]


# In[7]:

batches_labels = group_list(list(data.values()), 10)
lbls = np.zeros((10, 2))
num = 0
for batch in tqdm(batches_labels):
    num += 1
    for label, ind in zip(batch, range(10)):
        label_vector=np.zeros(2)
        label_vector[label]=1
        lbls[ind] = label_vector
    np.savez('/home/data2/vision6/azlu/training_data_10/label_tr_batch_{0}.npz'.format(num), labels=lbls)


# In[ ]:



