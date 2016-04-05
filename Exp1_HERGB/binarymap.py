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
from binmapdrop_pickling import *

srng = RandomStreams()

def create_binmap(learning_rate=0.01 , lr_dec = 0.4 , n_epochs=200 , nkerns=[50,80], batch_size=468):
    
    rng = np.random.RandomState(23455)

    testxdirname = 'Test/'
    testydirname = 'Test_anno/'
    testdataset = load_data(testxdirname,testydirname)

    n_binmap_batches = 468

    x = T.matrix('x')   
    y = T.ivector('y')
    
    print ('... building the model\n')

    layer0_input = x.reshape((batch_size , 5 , 51 , 51))

    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 5, 51, 51),
        filter_shape=(nkerns[0], , 9, 9),
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

    layer2_input = layer1.output.flatten(2)

    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 8 * 8,
        n_out=468,
        activation=T.tanh,
        dropout=0.5
    )

    layer3 = LogisticRegression(input=layer2.output, n_in=468, n_out=2)

    test_model2 = theano.function(
       [x , y],
       layer3.y_pred,
       allow_input_downcast = True,
       on_unused_input = 'ignore'
    )

    params = layer3.params + layer2.params + layer1.params + layer0.params

    print 'GETTING UPDATES'
    best_params = pickle.load(open( "weights.p", "rb" ))

    print 'READYING UPDATES............'
    best_updates = [
        (param_i,best_param_i)
        for param_i, best_param_i in zip(params,best_params)
    ]
    print 'UPDATES READY...............'
    best_model = theano.function(
        [],
        updates = best_updates,
        allow_input_downcast = True,
        on_unused_input = 'ignore'
    )
    print 'UPDATING BEST MODEL.........'
    best_model()
    print 'BEST MODEL UPDATED..........' 

    for im in range(0,72):
        print im
        nrow = testdataset[im][0].shape[0]
        for i in xrange(n_binmap_batches):
            binmap_x,binmap_y,ixgen,iygen = load_tl_img(i+25,testdataset[im])                 
            test_labels = test_model2(binmap_x , np.squeeze(np.asarray(binmap_y)))
            if i==0:
                binmap_xgen = np.asarray(ixgen)
                binmap_ygen = np.asarray(iygen)
                ypred = np.asarray(test_labels)
                yanno = np.squeeze(np.asarray(binmap_y))
            else:
                binmap_xgen = np.concatenate([binmap_xgen,np.asarray(ixgen)])
                binmap_ygen = np.concatenate([binmap_ygen,np.asarray(iygen)])
                ypred = np.concatenate([ypred,np.asarray(test_labels)])
                yanno = np.concatenate([yanno,np.squeeze(np.asarray(binmap_y))])                             
        binary_map(testdataset[im] , im , '_tl' , 468*468 , ypred , binmap_xgen , binmap_ygen)
        binary_map(testdataset[im] , im , '_tlbin' , 468*468 , yanno , binmap_xgen , binmap_ygen)

        
        for i in xrange(n_binmap_batches):
            binmap_x,binmap_y,ixgen,iygen = load_tr_img(nrow-i-26,testdataset[im])                 
            test_labels = test_model2(binmap_x , np.squeeze(np.asarray(binmap_y)))
            if i==0:
                binmap_xgen = np.asarray(ixgen)
                binmap_ygen = np.asarray(iygen)
                ypred = np.asarray(test_labels)
                yanno = np.squeeze(np.asarray(binmap_y))
            else:
                binmap_xgen = np.concatenate([binmap_xgen,np.asarray(ixgen)])
                binmap_ygen = np.concatenate([binmap_ygen,np.asarray(iygen)])
                ypred = np.concatenate([ypred,np.asarray(test_labels)])
                yanno = np.concatenate([yanno,np.squeeze(np.asarray(binmap_y))])                             
        binary_map(testdataset[im] , im , '_bl' , 468*468 , ypred , binmap_xgen , binmap_ygen)
        binary_map(testdataset[im] , im , '_blbin' , 468*468 , yanno , binmap_xgen , binmap_ygen)

if __name__ == '__main__':
    create_binmap()
