
# coding: utf-8

# In[91]:


from time import time
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
             discriminant_analysis, random_projection)



# In[92]:


## Loading and curating the data
digits = datasets.load_digits(n_class=10)
X = digits.data
y = digits.target
print X.shape
n_samples, n_features = X.shape
n_neighbors = 30
## Function to Scale and visualize the embedding vectors
# plt.scatter(1,1)
# plt.show()
# This code is used to make a plot with points denoting the number which is at which place in the 2 d graph which
# is embedded from 64 d graph . 


# In[93]:


def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
#     f=np.arange(1,11)
    X = (X - x_min)/(x_max-x_min)    
    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(digits.target[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
                 


# In[94]:


# Plot images of the digits
n_img_per_row = 20
#print X[1].reshape(8,8)

img = np.zeros((10 * n_img_per_row,  10 * n_img_per_row))
#print img.shape
for i in range(n_img_per_row):
    row = 10 * i + 1
    for j in range(n_img_per_row):
        col = 10 * j + 1
        img[row:row + 8, col:col + 8] = X[i * n_img_per_row + j].reshape((8, 8))
        # What we do in this loop is that we load the digit data from 10*i+j row in dataset and convert 1*64 to 8*8 
        # matrix and store it in img when we load img we get all the digits
       # print img[ix:ix + 8, iy:iy + 8]
plt.imshow(img, cmap=plt.cm.binary)
plt.xticks([])
plt.yticks([])
plt.title('A selection from the 64-dimensional digits dataset')


# In[95]:


## Computing PCA so as to compare its output with that of PCA Algorithm
print("Computing PCA projection")
t0 = time()
pca = decomposition.TruncatedSVD(n_components=2)
X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
plot_embedding(X_pca,
               "Principal Components projection of the digits (time %.2fs)" %
               (time() - t0))


# In[96]:


## Computing t-SNE
# manifold consists of large number of parameters namely :
# n_components which denote the number of parameters in embedding
# init:The embedding with which you want to initialize tsne with whether random or pca
# random_state: It is used as seed for pseudorandom number generator in scikit-learn to duplicate the 
# behavior when such randomness is involved in algorithms. Random_state is given in the machine Learning so 
# as to   produce the same output everytime the program is run
# Iterations:Number of iterations to be performed on 64 dimension data(Default:1000)
# fit _transform:Fit x into a state of 2 d and return that output


# In[97]:


print("Computing t-SNE embedding")
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
t0 = time()
X_tsne = tsne.fit_transform(X)

# X_tsne produces the 64 dimensional data in 2 dimension keeping the structure of the original structure almost 
# same by conserving the local structure
plot_embedding(X_tsne,
               "t-SNE embedding of the digits (time %.2fs)" %
               (time() - t0))
plt.show()





