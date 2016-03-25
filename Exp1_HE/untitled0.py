import os
#os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=gpu,floatX=float32'
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
def load_data(xdirname):
    print '...loading data'
    
    mat = []
    included_extensions = ['mat']
    filelist = [fn[:-6] for fn in os.listdir(xdirname) if any([fn.endswith(ext) for ext in included_extensions])]
    filelist = list(set(filelist))
    hext = '_H.mat'
    eext = '_E.mat'
    for fname in filelist:
        
        xname = xdirname + fname
        hname = xname + hext
        ename = xname + eext
        
        H = scipy.io.loadmat(hname)
        H = H['H']
        E = scipy.io.loadmat(ename)
        E = E['E']
#        
#        afname = fname[:-9] + '_bin.bmp'
#        yname = ydirname + afname
#        yim = Image.open(yname)
#        ypix = yim.load() 
#        
#        fgrnd = np.zeros((nrow,ncol))
        
#        for x in range(0, nrow):
#            for y in range(0,ncol):
#                red[x,y] = xpix[x,y][0]
#                green[x,y] = xpix[x,y][1]
#                blue[x,y] = xpix[x,y][2]
#                if ypix[x,y]:
#                    fgrnd[x,y] = 1.0

        img=np.array([H,E])
        mat.append(img)
        
    return mat

trainxdirname = '/media/rahul/1098D6BA98D69E12/academics/SEMESTER_7/BTP/Bachelor_Thesis_Project/Exp1_HE/Train/'

traindataset = load_data(trainxdirname)
print len(traindataset)
print traindataset[1][1].shape