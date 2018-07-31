"""
Compile MXNet Models
====================
**Author**: `Joshua Z. Zhang <https://zhreshold.github.io/>`_

This article is an introductory tutorial to deploy mxnet models with NNVM.

For us to begin with, mxnet module is required to be installed.

A quick solution is
```
pip install mxnet --user
```
or please refer to offical installation guide.
https://mxnet.incubator.apache.org/versions/master/install/index.html
"""
# some standard imports
import os, sys
thisdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(thisdir, '../mxnet-mobilenet-v2/tvm_local/python'))
sys.path.insert(0, os.path.join(thisdir, '../mxnet-mobilenet-v2/tvm_local/nnvm/python'))
sys.path.insert(0, os.path.join(thisdir, '../mxnet-mobilenet-v2/tvm_local/topi/python'))

import mxnet as mx
from mxnet import nd
import tvm
import nnvm
import numpy as np

num_inputs = 784
num_outputs = 10
num_hidden = 256
target = 'llvm'

def get_symbol(num_classes=10, **kwargs):
    data = mx.symbol.Variable('data')
    data = mx.sym.Flatten(data=data)
    fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=num_hidden)
    act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
    fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = num_hidden)
    act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
    fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=num_classes)
    mlp  = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')
    return mlp

######################################################################
# for a normal mxnet model, we start from here
# mx_sym, args, auxs = mx.model.load_checkpoint('resnet18_v1', 0)
mx_sym = get_symbol(num_outputs)
args = nd.load('models/mlp-%04d.params' % (19,))
args = {"fc1_weight": args[0].T, "fc1_bias": args[1],
        "fc2_weight": args[2].T, "fc2_bias": args[3], 
        "fc3_weight": args[4].T, "fc3_bias": args[5], }

# now we use the same API to get NNVM compatible symbol
nnvm_sym, nnvm_params = nnvm.frontend.from_mxnet(mx_sym, args, {})
# repeat the same steps to run this model using TVM

model_ctx = mx.cpu()
samples = 1
def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)
mnist_test = mx.gluon.data.vision.MNIST(train=False, transform=transform)

######################################################################
# now compile the graph
import nnvm.compiler
shape_dict = {'data': (1, 784)}
graph, lib, params = nnvm.compiler.build(nnvm_sym, target, shape_dict, params=nnvm_params)

# let's sample 10 random data points from the test set
sample_data = mx.gluon.data.DataLoader(mnist_test, samples, shuffle=True)
for i, (data, label) in enumerate(sample_data):
    data = data.as_in_context(model_ctx)
    im = nd.transpose(data,(1,0,2,3))
    im = nd.reshape(im,(28,samples*28,1))
    imtiles = nd.tile(im, (1,1,3))
    print('true labels :           ', int(label.asnumpy()[0]) )
    
    from mlp_predict import model_predict, net
    pred=model_predict(net,data.reshape((-1,784)))
    print('mxnet model prediction :', int(pred.asnumpy()[0]) )

    # Now, we would like to reproduce the same forward computation using TVM.
    from tvm.contrib import graph_runtime
    ctx = tvm.context(target, 0)
    dtype = 'float32'
    m = graph_runtime.create(graph, lib, ctx)
    # set inputs
    m.set_input('data', tvm.nd.array(data.asnumpy().astype(dtype)))
    m.set_input(**params)
    # execute
    m.run()
    # get outputs
    tvm_output = m.get_output(0, tvm.nd.empty((10,), dtype))
    top1 = np.argmax(tvm_output.asnumpy())
    print('TVM prediction top-1:   ', top1)

    print('--')
    if i > 10 : break
