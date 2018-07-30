
# ## Imports

# In[27]:


from __future__ import print_function
import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon


# ## Set contexts

# In[28]:


data_ctx = mx.cpu()
# model_ctx = mx.cpu()
model_ctx = mx.gpu(1)


# ## Load MNIST data
# 
# Let's go ahead and grab our data.

# In[29]:


num_inputs = 784
num_outputs = 10
batch_size = 64
num_examples = 60000
def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)
train_data = gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform),
                                      batch_size, shuffle=True)
test_data = gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                                     batch_size, shuffle=False)


# ## Allocate parameters
# 
# 

# In[30]:


#######################
#  Set some constants so it's easy to modify the network later
####################### 
num_hidden = 256
weight_scale = .01

params = nd.load('models/mlp-%04d.params' % (19,))
W1, b1, W2, b2, W3, b3 = params


# ## Activation functions
# 
# If we compose a multi-layer network but use only linear operations, then our entire network will still be a linear function. That's because $\hat{y} = X \cdot W_1 \cdot W_2 \cdot W_2 = X \cdot W_4 $ for $W_4 = W_1 \cdot W_2 \cdot W3$. To give our model the capacity to capture nonlinear functions, we'll need to interleave our linear operations with activation functions. In this case, we'll use the rectified linear unit (ReLU):

# In[32]:


def relu(X):
    return nd.maximum(X, nd.zeros_like(X))


# ## Softmax output
# 
# As with multiclass logistic regression, we'll want the outputs to constitute a valid probability distribution. We'll use the same softmax activation function on our output to make sure that our outputs sum to one and are non-negative.

# In[33]:


def softmax(y_linear):
    exp = nd.exp(y_linear-nd.max(y_linear))
    partition = nd.nansum(exp, axis=0, exclude=True).reshape((-1, 1))
    return exp / partition


# ## Define the model
# 
# Now we're ready to define our model

# In[36]:


def net(X):
    #######################
    #  Compute the first hidden layer
    #######################
    h1_linear = nd.dot(X, W1) + b1
    h1 = relu(h1_linear)

    #######################
    #  Compute the second hidden layer
    #######################
    h2_linear = nd.dot(h1, W2) + b2
    h2 = relu(h2_linear)

    #######################
    #  Compute the output layer.
    #  We will omit the softmax function here
    #  because it will be applied
    #  in the softmax_cross_entropy loss
    #######################
    yhat_linear = nd.dot(h2, W3) + b3
    return yhat_linear


# ## Using the model for prediction
# Let's pick a few random data points from the test set to visualize algonside our predictions. We already know quantitatively that the model is more accurate, but visualizing results is a good practice that can (1) help us to sanity check that our code is actually working and (2) provide intuition about what kinds of mistakes our model tends to make.

# In[ ]:


# get_ipython().magic(u'matplotlib inline')
# import matplotlib.pyplot as plt

# Define the function to do prediction
def model_predict(net,data):
    yhat_linear = net(data)
    output = softmax(yhat_linear)
    return nd.argmax(output, axis=1)

samples = 10

mnist_test = mx.gluon.data.vision.MNIST(train=False, transform=transform)

# let's sample 10 random data points from the test set
sample_data = mx.gluon.data.DataLoader(mnist_test, samples, shuffle=True)
for i, (data, label) in enumerate(sample_data):
    data = data.as_in_context(model_ctx)
    im = nd.transpose(data,(1,0,2,3))
    im = nd.reshape(im,(28,10*28,1))
    imtiles = nd.tile(im, (1,1,3))
    
    # plt.imshow(imtiles.asnumpy())
    # plt.show()
    pred=model_predict(net,data.reshape((-1,784)))
    print('model predictions are:', pred)
    print('true labels :', label)
    break


# ## Conclusion
# 
# Nice! With just two hidden layers containing 256 hidden nodes, respectively, we can achieve over 95% accuracy on this task. 

# ## Next
# [Multilayer perceptrons with gluon](../chapter03_deep-neural-networks/mlp-gluon.ipynb)

# For whinges or inquiries, [open an issue on  GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)
