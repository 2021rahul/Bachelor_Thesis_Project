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

def evaluate_lenet5(learning_rate=0.01 , lr_dec = 0.4 , n_epochs=200 , nkerns=[50,80], batch_size=390):
    
    rng = np.random.RandomState(23455)
    
    trainxdirname = 'Train/'
    trainydirname = 'Train_anno/'
    traindataset = load_data(trainxdirname,trainydirname)
    for vdatai in range(0,65):
        imgvx,imgvy,vxgen,vygen = load_img(360,traindataset[vdatai])
        if vdatai==0:
            validx = imgvx
            validy = imgvy
        else:
            validx = np.concatenate([validx,imgvx])
            validy = np.concatenate([validy,imgvy])
    print validx.shape
    print('valid dataset')
    print(sum(validy == 1))
    print(sum(validy == 0))


    testxdirname = 'Test/'
    testydirname = 'Test_anno/'
    testdataset = load_data(testxdirname,testydirname)
    txgen = []
    tygen = []
    for testi in range(0,13):
        imgvx,imgvy,xgen,ygen = load_img(1800,testdataset[testi])
        txgen.append(xgen)
        tygen.append(ygen)
        if testi==0:
            testx = imgvx
            testy = imgvy
        else:
            testx = np.concatenate([testx,imgvx])
            testy = np.concatenate([testy,imgvy])
    print testx.shape  
    print('test dataset')
    print(sum(testy == 1))
    print(sum(testy == 0))

    n_train_batches = 600
    n_valid_batches = validx.shape[0]/batch_size
    n_test_batches = testx.shape[0]/batch_size
    n_binmap_batches = 390

    x = T.matrix('x')   
    y = T.ivector('y')
    
    print ('... building the model\n')

    layer0_input = x.reshape((batch_size , 3 , 51 , 51))

    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 51, 51),
        filter_shape=(nkerns[0], 3, 9, 9),
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
        n_out=390,
        activation=T.tanh,
        dropout=0.5
    )

    layer3 = LogisticRegression(input=layer2.output, n_in=390, n_out=2)

    test_model2 = theano.function(
       [x , y],
       layer3.y_pred,
       allow_input_downcast = True,
       on_unused_input = 'ignore'
    )

    params = layer3.params + layer2.params + layer1.params + layer0.params

    print 'GETTING UPDATES'
    params = pickle.load(open( "weights.p", "wb" ))

    for im in range(0,13):
        for i in xrange(n_binmap_batches):
            binmap_x,binmap_y,ixgen,iygen = load_img3(i+25,testdataset[im])                 
            test_labels = test_model2(binmap_x , np.squeeze(np.asarray(binmap_y)))
            if i==0:
                binmap_xgen = np.asarray(ixgen)
                binmap_ygen = np.asarray(iygen)
                ypred = np.asarray(test_labels)
                #yanno = np.squeeze(np.asarray(binmap_y))
            else:
                binmap_xgen = np.concatenate([binmap_xgen,np.asarray(ixgen)])
                binmap_ygen = np.concatenate([binmap_ygen,np.asarray(iygen)])
                ypred = np.concatenate([ypred,np.asarray(test_labels)])
                #yanno = np.concatenate([yanno,np.squeeze(np.asarray(binmap_y))])                             
        binary_map(testdataset[im] , im , 390*390 , ypred , binmap_xgen , binmap_ygen)
        #binary_map(testdataset[im] , im*10 , 390*390 , yanno , binmap_xgen , binmap_ygen)

if __name__ == '__main__':
    evaluate_lenet5()
