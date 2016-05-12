import cPickle as pickle
import gzip
import os
import sys
import time
import PIL.Image
import cPickle
import scipy
import select
import csv

import numpy as np
import numpy
import scipy.io
from matplotlib import pyplot as plt
from sklearn.feature_extraction import image
from collections import OrderedDict

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer, SigmoidLayer, BinarizationLayer
from utils import tile_raster_images
from layers import ConvPoolLayer, LeNetConvPoolLayer
from patches import get_patches, reconstruct_image
#from tt_t import test_model

#from pylearn2.datasets.preprocessing import GlobalContrastNormalization


class CNN(object):
    """ convolutional neural network class"""
    def __init__(self, input, target, ishape, nkerns, batch_size):

        self.input = input
        self.target = target
        self.ishape = ishape
        self.batch_size = batch_size
        self.nkerns = nkerns
        rng = numpy.random.RandomState(23455)

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the model'

        # Reshape matrix of rasterized images of shape (batch_size,48*48)
        # to a 4D tensor, compatible with our ConvPoolLayer
        self.layer0_input = self.input.reshape((self.batch_size, 1, self.ishape[0], self.ishape[1]))

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (48-5+1,48-5+1)=(44,44)
        # maxpooling reduces this further to (44/2,44/2) = (22,22)
        # 4D output tensor is thus of shape (batch_size,nkerns[0],22,22)
        self.layer0 = ConvPoolLayer(rng, input=self.layer0_input,
                image_shape=(self.batch_size, 1, self.ishape[0], self.ishape[1]),
                filter_shape=(self.nkerns[0], 1, 15,15), poolsize=(2, 2))

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (22-5+1,22-5+1)=(18,18)
        # maxpooling reduces this further to (18/2,18/2) = (9,9)
        # 4D output tensor is thus of shape (nkerns[0],nkerns[1],9,9)
        self.layer1 = ConvPoolLayer(rng, input=self.layer0.output,
                image_shape=(self.batch_size, self.nkerns[0], 43, 43),
                filter_shape=(self.nkerns[1], self.nkerns[0], 10, 10), poolsize=(2, 2))

        self.layer1b = ConvPoolLayer(rng, input=self.layer1.output,
                image_shape=(self.batch_size, self.nkerns[1], 17, 17),
                filter_shape=(self.nkerns[1], self.nkerns[1], 4, 4), poolsize=(2, 2))
        # the TanhLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (20,32*9*9) = (20,2592)
        self.layer2_input = self.layer1b.output.flatten(2)

        # construct a fully-connected sigmoidal layer
        self.layer2 = HiddenLayer(rng, input=self.layer2_input, n_in=self.nkerns[1] * 7 * 7,
                             n_out=1000, activation=T.tanh)

        # classify the values of the fully-connected sigmoidal layer
        self.layer3 = BinarizationLayer(rng, input=self.layer2.output, n_in=1000, n_out=50*50, activation=T.nnet.sigmoid)

        self.params = self.layer3.params  + self.layer2.params  + self.layer1b.params  + self.layer1.params  + self.layer0.params
        
        #self.cost = self.layer3.xent(self.target)
        

        # for every parameter, we maintain it's last update
        # the idea here is to use "momentum"
        # keep moving mostly in the same direction
        self.updates = {}
        for param in self.params:
            init = np.zeros(param.get_value(borrow=True).shape,
                            dtype=theano.config.floatX)
            self.updates[param] = theano.shared(init)
            
    def Cost(self, y):
        return self.layer3.xent(y)
        
    def Errors(self, y):
        return self.layer3.errors(y)
        
    def acc_model(self, y):
        return [self.layer3.sensitivity(y), self.layer3.specificity(y)]
        
    
class MultiScaleCNN(object):
    """ convolutional neural network class"""
    def __init__(self, input, target, ishape, nkerns, batch_size):

        self.input = input
        self.target = target
        self.ishape = ishape
        self.batch_size = batch_size
        self.nkerns = nkerns
        rng = numpy.random.RandomState(23455)

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the model'


        # Reshape matrix of rasterized images of shape (batch_size,48*48)
        # to a 4D tensor, compatible with our ConvPoolLayer
        self.layer0_input = self.input.reshape((self.batch_size, 1, self.ishape[0], self.ishape[1]))

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (48-5+1,48-5+1)=(44,44)
        # maxpooling reduces this further to (44/2,44/2) = (22,22)
        # 4D output tensor is thus of shape (batch_size,nkerns[0],22,22)
        self.layer0 = LeNetConvPoolLayer(rng, input=self.layer0_input,
                image_shape=(self.batch_size, 1, self.ishape[0], self.ishape[1]),
                filter_shape=(self.nkerns[0], 1, 7,7), poolsize=(2, 2))

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (22-5+1,22-5+1)=(18,18)
        # maxpooling reduces this further to (18/2,18/2) = (9,9)
        # 4D output tensor is thus of shape (nkerns[0],nkerns[1],9,9)
        self.layer1 = LeNetConvPoolLayer(rng, input=self.layer0.output,
                image_shape=(self.batch_size, self.nkerns[0], 47, 47),
                filter_shape=(self.nkerns[1], self.nkerns[0], 6, 6), poolsize=(2, 2))

        self.layer1b = LeNetConvPoolLayer(rng, input=self.layer1.output,
                image_shape=(self.batch_size, self.nkerns[1], 21, 21),
                filter_shape=(self.nkerns[1], self.nkerns[1], 6, 6), poolsize=(2, 2))
        # the TanhLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (20,32*9*9) = (20,2592)
        self.layer2_input = self.layer1b.output.flatten(2)

        # construct a fully-connected sigmoidal layer
        self.layer2 = HiddenLayer(rng, input=self.layer2_input, n_in=self.nkerns[1] * 8 * 8,
                             n_out=2000, activation=T.nnet.sigmoid)

        # classify the values of the fully-connected sigmoidal layer
        self.layer3 = BinarizationLayer(rng, input=self.layer2.output, n_in=2000, n_out=50*50, activation=T.nnet.sigmoid)

        self.params = self.layer3.params  + self.layer2.params  + self.layer1b.params  + self.layer1.params  + self.layer0.params
        
        #self.cost = self.layer3.xent(self.target)
        

        # for every parameter, we maintain it's last update
        # the idea here is to use "momentum"
        # keep moving mostly in the same direction
        self.updates = {}
        for param in self.params:
            init = np.zeros(param.get_value(borrow=True).shape,
                            dtype=theano.config.floatX)
            self.updates[param] = theano.shared(init)
            
    def Cost(self, y):
        return self.layer3.xent(y)
        
    def Errors(self, y):
        return self.layer3.errors(y)
        
    def acc_model(self, y):
        return [self.layer3.sensitivity(y), self.layer3.specificity(y)]   



def ncc( Data, CC = 100):
    Data = Data/255
    datamean = np.mean(Data, axis = 1)
    Data = Data - np.tile(np.reshape(np.mean(Data,axis=1),(Data.shape[0],1)),(1,Data.shape[1]))
    datanorm = np.sqrt(np.sum(np.multiply(Data,Data),1))
    normeq0 = datanorm < 1e-8
    if any(normeq0):
        datanorm[normeq0] = 1
    Data = CC * Data / np.tile(np.reshape(datanorm,(Data.shape[0],1)),( 1, Data.shape[1]))
    return Data

def evaluate_net(learning_rate=0.5, n_epochs=1,
                    nkerns=[20, 50], batch_size=100):

    print(" -------------Python programme for blood vessel segmentation--------------- ")
    print(" conv net with three layers, 100*100 image size with 0.5 momentum")
    print(" conv net with 0.5 lr(decay)")
    print(" conv net with btach_size 400")
    rng = numpy.random.RandomState(23455)
    final_momentum = 0.5
    initial_momentum = 0.9
    momentum_switchover = 20
    learning_rate_decay = 0.998

    ########### training set ############################
    DataX = np.zeros((3105*9,100,100))
    DataY = np.zeros((3105*9,50,50))
    name = 'dr','g','h'
    num_patches = 3105
    import os
    current_dir = os.getcwd()
    
    for i in xrange(3):
        img = plt.imread(current_dir+'/HRF/all/images/'+str(i+1).zfill(2)+'_dr.JPG')
        mask = plt.imread(current_dir+'/HRF/all/mask/'+str(i+1).zfill(2)+'_dr_mask.tif')
        gt = plt.imread(current_dir+'/HRF/all/manual1/'+str(i+1).zfill(2)+'_dr.tif')
        mask =mask[:,:,1]
        img = img[:,:,1]
        #img =img*mask
        patches = get_patches(img, (100,100), padding=0, overlap=1)
        DataX[3*i*num_patches:3*i*num_patches+num_patches,:,:]=patches
        patches = get_patches(gt, (100,100), padding=0, overlap=1)
        DataY[3*i*num_patches:3*i*num_patches+num_patches,:,:]=patches[:,25:75,25:75]
        
        img = plt.imread(current_dir+'/HRF/all/images/'+str(i+1).zfill(2)+'_g.jpg')
        mask = plt.imread(current_dir+'/HRF/all/mask/'+str(i+1).zfill(2)+'_g_mask.tif')
        gt = plt.imread(current_dir+'/HRF/all/manual1/'+str(i+1).zfill(2)+'_g.tif')
        mask =mask[:,:,1]
        img = img[:,:,1]
        #img =img*mask
        patches = get_patches(img, (100,100), padding=0, overlap=1)
        DataX[3*i*num_patches+num_patches:3*i*num_patches+2*num_patches,:,:]=patches
        patches = get_patches(gt, (100,100), padding=0, overlap=1)
        DataY[3*i*num_patches+num_patches:3*i*num_patches+2*num_patches,:,:]=patches[:,25:75,25:75]
        
        img = plt.imread(current_dir+'/HRF/all/images/'+str(i+1).zfill(2)+'_h.jpg')
        mask = plt.imread(current_dir+'/HRF/all/mask/'+str(i+1).zfill(2)+'_h_mask.tif')
        gt = plt.imread(current_dir+'/HRF/all/manual1/'+str(i+1).zfill(2)+'_h.tif')
        mask =mask[:,:,1]
        img = img[:,:,1]
        #img =img*mask
        patches = get_patches(img, (100,100), padding=0, overlap=1)
        DataX[3*i*num_patches+2*num_patches:3*i*num_patches+3*num_patches,:,:]=patches
        patches = get_patches(gt, (100,100), padding=0, overlap=1)
        DataY[3*i*num_patches+2*num_patches:3*i*num_patches+3*num_patches,:,:]=patches[:,25:75,25:75]

    #DataY = DataY[:,25:75,25:75]
    DataX = np.reshape(DataX, (3105*9, 100*100))
    DataY = np.reshape(DataY, (3105*9, 50*50))
    CC = 150
    #DataX = ncc(DataX, CC)
    #prepro.apply(DataX)
    DataY = DataY/np.max(DataY)
    assert np.max(DataY) == 1
 
    train_set_y = DataY[:7*3105,]
    valid_set_y = DataY[7*3105:,]   
    train_set_x = np.asarray(DataX[:7*3105,:])
    valid_set_x = DataX[7*3105:,:]
    #train_set_x = ncc(train_set_x, CC)
    #valid_set_x = ncc(valid_set_x, CC)
    
    inds = np.random.permutation(train_set_x.shape[0])
    train_set_x = train_set_x[inds[:],]
    train_set_y = train_set_y[inds[:],]
    inds = np.random.permutation(valid_set_x.shape[0])
    valid_set_x = valid_set_x[inds[:],]
    valid_set_y = valid_set_y[inds[:],]
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.shape[0]
    n_valid_batches = valid_set_x.shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size


    train_set_x = T.cast(theano.shared(numpy.asarray(train_set_x)), 'float32')
    valid_set_x = T.cast(theano.shared(numpy.asarray(valid_set_x)), 'float32')
    train_set_y = T.cast(theano.shared(numpy.asarray(train_set_y)), 'int32')
    valid_set_y = T.cast(theano.shared(numpy.asarray(valid_set_y)), 'int32')

    '''#######################
    #### testing ####
    #######################
    img = plt.imread('../HRF/all/images/'+str(6).zfill(2)+'_g.jpg')
    mask = plt.imread('../HRF/all/mask/'+str(6).zfill(2)+'_g_mask.tif')
    gt = plt.imread('../HRF/all/manual1/'+str(6).zfill(2)+'_g.tif')
    mask =mask[:,:,1]
    img = img[:,:,1]
    img =img*mask
    patches = get_patches(img, (100,100), overlap=1)
    DataZ=patches
    
    num_patches1 = DataZ.shape[0]
    DataZ = np.reshape(DataZ, (num_patches1, 100*100), overlap=1)

    CC = 150
    DataZ = ncc(DataZ, CC)
    test_set_x = np.asarray(DataZ)
    
    # compute number of minibatches for training, validation and testing
    n_test_batches = test_set_x.shape[0]
    n_test_batches /= batch_size
    test_set_x = T.cast(theano.shared(numpy.asarray(test_set_x)), 'float64')  
    ########################'''
          
    # allocate symbolic variables for the data
    index = T.lscalar()
    x = T.fmatrix('x')   
    y = T.imatrix('y')  
    ishape = (100, 100)  # this is the size of face images


    cnn = MultiScaleCNN(input = x, target = y, ishape = ishape, nkerns = nkerns, batch_size = batch_size)
    # the cost we minimize during training is the NLL of the model
    cost = cnn.Cost(y)
    #cost = layer3.xent(y)
    print_cost = theano.function([index], cost,
          givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size], 
            y: train_set_y[index * batch_size: (index + 1) * batch_size]})
   
    # create a function to compute the mistakes that are made by the model
    validate_model = theano.function([index], cnn.Errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]})

    accuracy_model = theano.function([index], cnn.acc_model(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]})
                
    '''output = theano.function([index],cnn.layer3.output,givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size]})  '''

    
    mom = T.scalar('mom', dtype=theano.config.floatX)
    l_r = T.scalar('l_r', dtype=theano.config.floatX)
    grads = T.grad(cost = cost, wrt = cnn.params)
    '''gparams = []
    for param in cnn.params:
        gparam = T.grad(cost = cost, wrt = param)
        gparams.append(gparam)'''

    updates = []
    for param, gparam in zip(cnn.params, grads):
        weight_update = cnn.updates[param]
        upd = mom * weight_update - l_r * gparam
        updates.append((cnn.updates[param], upd))
        updates.append((param, param + upd))

        
        
    train_model = theano.function([index, l_r, mom], cost, updates=updates,
      givens={
        x: train_set_x[index * batch_size: (index + 1) * batch_size],
        y: train_set_y[index * batch_size: (index + 1) * batch_size]})
    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 30000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    best_validation_sensitivity = 0
    best_validation_specificity = 0
    best_iter = 0
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index
            momentum = final_momentum \
                           if epoch > momentum_switchover \
                           else initial_momentum

            if iter % 100 == 0:
                print 'training @ iter = ', iter
                
            cost_ij = train_model(minibatch_index, learning_rate, momentum)
            print print_cost(minibatch_index)
            
            '''############
            ### test ###
            output_patches = np.zeros(DataZ.shape)
            for i in xrange(n_test_batches):
                output_patches[i*batch_size:(i+1)*batch_size,:] = output(i)
            output_patches = np.reshape(output_patches,(output_patches.shape[0],100,100))
            img_op = reconstruct_image(output_patches, img.shape)
            img_op = img_op * (mask/255)
            plt.imsave('../Images/vessel_at_epoch_%i.png' %epoch,img_op>0.65, cmap = plt.cm.gray, format = 'png')
            ############'''
            
            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation accuracy %f %%' % \
                      (epoch, minibatch_index + 1, n_train_batches, \
                       100 - this_validation_loss * 100.))

                validation_evaluation = [accuracy_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_evaluation = numpy.mean(validation_evaluation, axis=0)
                print('epoch %i, minibatch %i/%i, validation sensitivity %f %%' % \
                      (epoch, minibatch_index + 1, n_train_batches, \
                       this_validation_evaluation[0] * 100.))
                print('epoch %i, minibatch %i/%i, validation specificicity %f %%' % \
                      (epoch, minibatch_index + 1, n_train_batches, \
                       this_validation_evaluation[1] * 100.))
                       
                       
                # if we got the best validation score until now
                if this_validation_evaluation[0] > best_validation_sensitivity or \
                        best_validation_loss > this_validation_loss or \
                        best_validation_specificity < this_validation_evaluation[1] :
                
                    if this_validation_loss < 0.2:
                        params0 = cnn.layer0.params
                        params1 = cnn.layer1.params
                        params1b = cnn.layer1b.params
                        params2 = cnn.layer2.params
                        params3 = cnn.layer3.params
                        model = { 'layer0' : params0, 'layer1' : params1, \
                                'layer1b' : params1b, 'layer2' : params2, 'layer3' : params3}
                        f = open( '../model/expt/tt_overlap_mod_' + \
                                str(int(100-this_validation_loss*100)) + '__' + \
                                str(int(100*this_validation_evaluation[0]))+ '__' + \
                                str(int(100*this_validation_evaluation[1]))+ '.p' ,'wb')
                        pickle.dump(model, f, -1)
                        f.close()

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_validation_sensitivity = this_validation_evaluation[0] 
                    best_validation_specificity = this_validation_evaluation[1]
                    best_iter = iter

            learning_rate *= learning_rate_decay
            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i,' %(best_validation_loss * 100., best_iter + 1))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))



if __name__ == '__main__':
    print "chutiyapa continues..."
    evaluate_net()
