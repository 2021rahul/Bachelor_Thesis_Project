import os
os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=gpu,floatX=float32'
import numpy as np
import theano
import theano.tensor as T
import pickle
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import random
from PIL import Image
import scipy.io
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from binmapdrop_pickling_multiscale import *

learning_rate=0.01
lr_dec = 0.4
n_epochs=200
nkerns=[50,80]
batch_size=468
rng = np.random.RandomState(23455)

trainxdirname = 'Train/'
trainydirname = 'Train_anno/'
traindataset = load_data(trainxdirname,trainydirname)
for datai in range(0,78):
	imgvx,imgvy,vxgen,vygen = load_img(6,traindataset[datai])
	if datai==0:
		trainx = imgvx
		trainy = imgvy
	else:
		trainx = np.concatenate([trainx,imgvx])
		trainy = np.concatenate([trainy,imgvy])

x = T.matrix('x')   
y = T.ivector('y')

layer0_input = x.reshape((batch_size , 5 , 51 , 51))

layer0 = LeNetConvPoolLayer(
    rng,
    input=layer0_input,
    image_shape=(batch_size, 5, 51, 51),
    filter_shape=(nkerns[0], 5, 9, 9),
    dropout = 0.1,
    poolsize=(2 , 2)  
)

layer1 = LeNetConvPoolLayer(
    rng,
    input=layer0.output,
    image_shape=(batch_size , nkerns[0] , 21 , 21),
    filter_shape=(nkerns[1] , nkerns[0] , 5 , 5),
    dropout = 0.1,
    poolsize=(2 , 2)
)

layer0_out = theano.function(
	[x,y],
	layer0.output,
	on_unused_input='ignore',
	allow_input_downcast=True
)

layer1_out = theano.function(
	[x,y],
	layer1.output,
	on_unused_input='ignore',
	allow_input_downcast=True
)


out0 = layer0_out(trainx,np.squeeze(np.asarray(trainy)))
out1 = layer1_out(trainx,np.squeeze(np.asarray(trainy)))

print type(out0)
print type(out1)
print out0.shape
print out1.shape

print out0.flatten(2).shape
print out1.flatten(2).shape

new = T.concatenate([out0.flatten(2),out1.flatten(2)])

