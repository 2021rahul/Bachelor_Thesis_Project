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

srng = RandomStreams()
def load_data(xdirname , ydirname):
    print '...loading data'
    
    mat = []
    
    included_extensions = ['bmp']
    filelist = [fn for fn in os.listdir(xdirname) if any([fn.endswith(ext) for ext in included_extensions])]
    
    for fname in filelist:
        
        xname = xdirname + fname
        xim = Image.open(xname)
        xpix = xim.load()       
        
        nrow = xim.size[0]
        ncol = xim.size[1]    
        
        red = np.zeros((nrow,ncol))
        green = np.zeros((nrow,ncol))
        blue = np.zeros((nrow,ncol))
        
        afname = fname[:-9] + '_bin.bmp'
        yname = ydirname + afname
        yim = Image.open(yname)
        ypix = yim.load() 
        
        fgrnd = np.zeros((nrow,ncol))
        
        for x in range(0, nrow):
            for y in range(0,ncol):
                red[x,y] = xpix[x,y][0]
                green[x,y] = xpix[x,y][1]
                blue[x,y] = xpix[x,y][2]
                if ypix[x,y]:
                    fgrnd[x,y] = 1.0

        img=np.array([red,green,blue,fgrnd])
        mat.append(img)
        
    return mat
    
def load_img(num,img):
    yval = np.zeros((1,1))
    num0 = 0
    num1 = 0
    nrow = img[0].shape[0]
    ncol = img[0].shape[1]
    xgen = []
    ygen = []    
    while num0<num/2 or num1<num/2:
        
        x = random.randint(25,nrow-26)
        y = random.randint(25,ncol-26)
            
        if(
            img[3][x,y]==0 and num0<num/2 or
            img[3][x,y]==1 and num1<num/2
        ):
         
            xred = img[0][x-25:x+26,y-25:y+26]
            xgreen = img[1][x-25:x+26,y-25:y+26]
            xblue = img[2][x-25:x+26,y-25:y+26]
            yval[0][0] = img[3][x,y]
            
            xred = xred.reshape((1,2601))
            xgreen = xgreen.reshape((1,2601))
            xblue = xblue.reshape((1,2601))
        
            ximg = np.concatenate([xred , xgreen , xblue] , axis=1)
            xgen.append(x)
            ygen.append(y)
            if num0==0 and num1==0:
                datax = ximg
                datay = yval
            else:
                datax = np.concatenate([datax,ximg])
                datay = np.concatenate([datay,yval])
                
            if img[3][x,y]:
                num1 = num1+1
            else:
                num0 = num0+1
                                         
    rval = [datax,datay,xgen,ygen]
    return rval

def load_tl_img(x,img):
    yval = np.zeros((1,1))
    nrow = img[0].shape[0]
    ncol = img[0].shape[1]
    xgen = []
    ygen = []    
    
    for y in range(25,468+25):
        xred = img[0][x-25:x+26,y-25:y+26]
        xgreen = img[1][x-25:x+26,y-25:y+26]
        xblue = img[2][x-25:x+26,y-25:y+26]
        yval[0][0] = img[3][x,y]
        
        xred = xred.reshape((1,2601))
        xgreen = xgreen.reshape((1,2601))
        xblue = xblue.reshape((1,2601))
    
        ximg = np.concatenate([xred , xgreen , xblue] , axis=1)
        xgen.append(x)
        ygen.append(y)
        if y==25:
            datax = ximg
            datay = yval
        else:
            datax = np.concatenate([datax,ximg])
            datay = np.concatenate([datay,yval])
                                         
    rval = [datax,datay,xgen,ygen]
    return rval
    
def load_tr_img(x,img):
    yval = np.zeros((1,1))
    nrow = img[0].shape[0]
    ncol = img[0].shape[1]
    xgen = []
    ygen = []    
    
    for y in range(ncol-25-468,ncol-25):
        xred = img[0][x-25:x+26,y-25:y+26]
        xgreen = img[1][x-25:x+26,y-25:y+26]
        xblue = img[2][x-25:x+26,y-25:y+26]
        yval[0][0] = img[3][x,y]
        
        xred = xred.reshape((1,2601))
        xgreen = xgreen.reshape((1,2601))
        xblue = xblue.reshape((1,2601))
    
        ximg = np.concatenate([xred , xgreen , xblue] , axis=1)
        xgen.append(x)
        ygen.append(y)
        if y==ncol-25-390:
            datax = ximg
            datay = yval
        else:
            datax = np.concatenate([datax,ximg])
            datay = np.concatenate([datay,yval])
                                         
    rval = [datax,datay,xgen,ygen]
    return rval
    
def load_bl_img(x,img):
    yval = np.zeros((1,1))
    nrow = img[0].shape[0]
    ncol = img[0].shape[1]
    xgen = []
    ygen = []    
    
    for y in range(25,468+25):
        xred = img[0][x-25:x+26,y-25:y+26]
        xgreen = img[1][x-25:x+26,y-25:y+26]
        xblue = img[2][x-25:x+26,y-25:y+26]
        yval[0][0] = img[3][x,y]
        
        xred = xred.reshape((1,2601))
        xgreen = xgreen.reshape((1,2601))
        xblue = xblue.reshape((1,2601))
    
        ximg = np.concatenate([xred , xgreen , xblue] , axis=1)
        xgen.append(x)
        ygen.append(y)
        if y==25:
            datax = ximg
            datay = yval
        else:
            datax = np.concatenate([datax,ximg])
            datay = np.concatenate([datay,yval])
                                         
    rval = [datax,datay,xgen,ygen]
    return rval
    
    
def load_br_img(x,img):
    yval = np.zeros((1,1))
    nrow = img[0].shape[0]
    ncol = img[0].shape[1]
    xgen = []
    ygen = []    
    
    for y in range(ncol-25-468,ncol-25):
        xred = img[0][x-25:x+26,y-25:y+26]
        xgreen = img[1][x-25:x+26,y-25:y+26]
        xblue = img[2][x-25:x+26,y-25:y+26]
        yval[0][0] = img[3][x,y]
        
        xred = xred.reshape((1,2601))
        xgreen = xgreen.reshape((1,2601))
        xblue = xblue.reshape((1,2601))
    
        ximg = np.concatenate([xred , xgreen , xblue] , axis=1)
        xgen.append(x)
        ygen.append(y)
        if y==ncol-25-390:
            datax = ximg
            datay = yval
        else:
            datax = np.concatenate([datax,ximg])
            datay = np.concatenate([datay,yval])
                                         
    rval = [datax,datay,xgen,ygen]
    return rval

def load_img3(x,img):
    yval = np.zeros((1,1))
    nrow = img[0].shape[0]
    ncol = img[0].shape[1]
    xgen = []
    ygen = []    

    for y in range(25,415):
        xred = img[0][x-25:x+26,y-25:y+26]
        xgreen = img[1][x-25:x+26,y-25:y+26]
        xblue = img[2][x-25:x+26,y-25:y+26]
        yval[0][0] = img[3][x,y]
        
        xred = xred.reshape((1,2601))
        xgreen = xgreen.reshape((1,2601))
        xblue = xblue.reshape((1,2601))
    
        ximg = np.concatenate([xred , xgreen , xblue] , axis=1)
        xgen.append(x)
        ygen.append(y)
        if y==25:
            datax = ximg
            datay = yval
        else:
            datax = np.concatenate([datax,ximg])
            datay = np.concatenate([datay,yval])
                                         
    rval = [datax,datay,xgen,ygen]
    return rval

def binary_map(img , img_i , lab , num , ypred , x , y):
    nrow = img[0].shape[0]
    ncol = img[0].shape[1]
    bin_map = np.zeros((nrow,ncol))
    print len(x)
    print len(y)
    print len(ypred)
    i=0
    while i<num:
        bin_map[x[i]][y[i]] = ypred[i]
        i += 1

    bin_name = 'maps/' + str(img_i) + lab + '.mat'
    print bin_name
    scipy.io.savemat(bin_name,mdict={'bin_map': bin_map})
    
class LeNetConvPoolLayer(object):

    def __init__(self, rng, input, filter_shape,image_shape, dropout,poolsize=(2, 2)):
        assert image_shape[1] == filter_shape[1]
        self.input = input

        fan_in = np.prod(filter_shape[1:])
        
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(poolsize))
        
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        if dropout > 0.0:
            retain_prob = 1 - dropout
            pooled_out *= srng.binomial(pooled_out.shape, p=retain_prob, dtype=theano.config.floatX)
            pooled_out /= retain_prob

        self.output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        self.output = theano.tensor.switch(self.output<0, 0, self.output)
        self.params = [self.W, self.b]
        self.input = input
        
        
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out,dropout, W=None, b=None,activation=T.tanh):

        self.input = input
        
        if W is None:
            W_values = np.asarray(rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b

        if activation is 'relu': 
            out = (theano.tensor.switch(lin_output<0, 0, lin_output))
        else:
            out = T.tanh(lin_output)                
                       
        if dropout > 0.0:
            srng = theano.tensor.shared_randomstreams.RandomStreams(
                rng.randint(999999))
            mask = srng.binomial(n=1, p=1-dropout, size=out.shape)
            self.output = out * T.cast(mask, theano.config.floatX)
        else:
            self.output = out
        self.params = [self.W, self.b]
        
class LogisticRegression(object):
    
    def __init__(self, input, n_in, n_out, W = None, b = None):

        if W is None:
                W = theano.shared(value=np.zeros((n_in, n_out),dtype=theano.config.floatX),name='W', borrow=True)

        if b is None:
                b = theano.shared(value=np.zeros((n_out,), dtype=theano.config.floatX), name='b', borrow=True)
       
        self.W = W
        self.b = b

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type))
                
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
        print('here \n')
        
def evaluate_lenet5(learning_rate=0.01 , lr_dec = 0.4 , n_epochs=200 , nkerns=[50,80], batch_size=468):
    
    rng = np.random.RandomState(23455)
    
    trainxdirname = 'Train/'
    trainydirname = 'Train_anno/'
    traindataset = load_data(trainxdirname,trainydirname)
    for vdatai in range(0,78):
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


    # testxdirname = '/home/amit-pc/BTP_ashima/Bachelor_Thesis_Project/total_data_normalized/Test/'
    # testydirname = '/home/amit-pc/BTP_ashima/Bachelor_Thesis_Project/total_data_normalized/Test_anno/'
    # testdataset = load_data(testxdirname,testydirname)
    # txgen = []
    # tygen = []
    # for testi in range(0,13):
    #     imgvx,imgvy,xgen,ygen = load_img(1800,testdataset[testi])
    #     txgen.append(xgen)
    #     tygen.append(ygen)
    #     if testi==0:
    #         testx = imgvx
    #         testy = imgvy
    #     else:
    #         testx = np.concatenate([testx,imgvx])
    #         testy = np.concatenate([testy,imgvy])
    # print testx.shape  
    # print('test dataset')
    # print(sum(testy == 1))
    # print(sum(testy == 0))

    n_train_batches = 600
    n_valid_batches = validx.shape[0]/batch_size
    # n_test_batches = testx.shape[0]/batch_size
    # n_binmap_batches = 468

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
        n_out=468,
        activation=T.tanh,
        dropout=0.5
    )

    layer3 = LogisticRegression(input=layer2.output, n_in=468, n_out=2)

    cost = layer3.negative_log_likelihood(y)

    validate_model = theano.function(
        [x , y],
        layer3.errors(y),
        allow_input_downcast=True
    )

    test_model = theano.function(
        [x , y],
        layer3.errors(y),
        allow_input_downcast=True
    )

    test_model2 = theano.function(
	   [x , y],
	   layer3.y_pred,
	   allow_input_downcast = True,
	   on_unused_input = 'ignore'
    )

    params = layer3.params + layer2.params + layer1.params + layer0.params
    best_params = params
    grads = T.grad(cost, params)

    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ] 

    train_model = theano.function(
        [x , y],
        cost,
        updates=updates,
        allow_input_downcast=True
    )

    print '... training'
    
    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience / 2)
    best_validation_loss = np.inf
    best_test_loss = np.inf
    best_validation_iter = 0
    best_test_iter = 0

    epoch = 0
    done_looping = False    

    while (epoch < n_epochs) and (not done_looping):
        
        epoch = epoch + 1

        for minibatch_index in xrange(n_train_batches):

            for datai in range(0,78):
                imgvx,imgvy,vxgen,vygen = load_img(6,traindataset[datai])
                if datai==0:
                    trainx = imgvx
                    trainy = imgvy
                else:
                    trainx = np.concatenate([trainx,imgvx])
                    trainy = np.concatenate([trainy,imgvy])

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print ('training @ iter = %i\n'%iter)
		
            cost_ij = train_model(trainx,np.squeeze(np.asarray(trainy)))

            if (iter + 1) % validation_frequency == 0:

                validation_losses = [
                    validate_model(validx[i*batch_size:(i+1)*batch_size] , np.squeeze(np.asarray(validy[i*batch_size:(i+1)*batch_size]))) 
                    for i in xrange(n_valid_batches)
                ]
                                     
                this_validation_loss = np.mean(validation_losses)

                print ('epoch %i, minibatch %i/%i, validation error %f %%\n' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))
                       
                # test_losses = [
                #     test_model(testx[i*batch_size:(i+1)*batch_size] , np.squeeze(np.asarray(testy[i*batch_size:(i+1)*batch_size])))
                #     for i in xrange(n_test_batches)
                # ]
                # this_test_score = np.mean(test_losses)
                # print (('test error of best model %f %%\n') %
                #         (this_test_score * 100.))

                # if this_test_score < best_test_loss:
                #     best_test_loss = this_test_score
                #     best_test_iter = iter
                #     best_params = params

                if this_validation_loss < best_validation_loss:

                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                       patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_validation_iter = iter
                    best_params = params

            if patience <= iter:
                done_looping = True
                break
            learning_rate -= lr_dec*learning_rate

    print ('Best validation score of %f %% obtained at iteration %i %%\n' %
           (best_validation_loss * 100., best_validation_iter + 1))
           
    print ('Best test score of %f %% obtained at iteration %i %%\n' %
           (best_test_loss * 100., best_test_iter + 1))

    print 'SAVING UPDATES'
    pickle.dump(best_params, open( "weights.p", "wb" ) )
    # print 'READYING UPDATES............'
    # best_updates = [
    #     (param_i,best_param_i)
    #     for param_i, best_param_i in zip(params,best_params)
    # ]
    # print 'UPDATES READY...............'
    # best_model = theano.function(
    #     [],
    #     updates = best_updates,
    #     allow_input_downcast = True,
    #     on_unused_input = 'ignore'
    # )
    # print 'UPDATING BEST MODEL.........'
    # best_model()
    # print 'BEST MODEL UPDATED..........' 
    # for im in range(0,13):
    #     for i in xrange(n_binmap_batches):
    #         binmap_x,binmap_y,ixgen,iygen = load_img3(i+25,testdataset[im])                 
    #         test_labels = test_model2(binmap_x , np.squeeze(np.asarray(binmap_y)))
    #         if i==0:
    #             binmap_xgen = np.asarray(ixgen)
    #             binmap_ygen = np.asarray(iygen)
    #             ypred = np.asarray(test_labels)
    #             #yanno = np.squeeze(np.asarray(binmap_y))
    #         else:
    #             binmap_xgen = np.concatenate([binmap_xgen,np.asarray(ixgen)])
    #             binmap_ygen = np.concatenate([binmap_ygen,np.asarray(iygen)])
    #             ypred = np.concatenate([ypred,np.asarray(test_labels)])
    #             #yanno = np.concatenate([yanno,np.squeeze(np.asarray(binmap_y))])                             
    #     binary_map(testdataset[im] , im , 390*390 , ypred , binmap_xgen , binmap_ygen)
    #     #binary_map(testdataset[im] , im*10 , 390*390 , yanno , binmap_xgen , binmap_ygen)

if __name__ == '__main__':
    evaluate_lenet5()
