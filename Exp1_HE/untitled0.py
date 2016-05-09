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
def load_data(xdirname , ydirname):
    print '...loading data'
    
    mat = []
    
    included_extensions = ['bmp']
    filelist = [fn[:-9] for fn in os.listdir(xdirname) if any([fn.endswith(ext) for ext in included_extensions])]
    iext = '_norm.bmp'
    hext = '_H.mat'
    eext = '_E.mat'
    aext = '_bin.bmp'
    
    for fname in filelist:
        
        xname = xdirname + fname
        iname = xname + iext
        hname = xname + hext
        ename = xname + eext
        aname = ydirname + fname + aext

        H = scipy.io.loadmat(hname)
        H = H['H']
        E = scipy.io.loadmat(ename)
        E = E['E']

        xim = Image.open(xname)
        xpix = xim.load()
        yim = Image.open(aname)
        ypix = yim.load()

        width,height = yim.size
        red = np.zeros((height,width))
        green = np.zeros((height,width))
        blue = np.zeros((height,width))
        fgrnd = np.zeros((height,width))
        
        for x in range(width):
            for y in range(height):
                red[y,x] = xpix[x,y][0]
                green[y,x] = xpix[x,y][1]
                blue[y,x] = xpix[x,y][2]
                if ypix[x,y]:
                    fgrnd[y,x] = 1.0

        img=np.array([red,green,blue,H,E,fgrnd])
        mat.append(img)
        
    return mat
    
trainxdirname = '/media/rahul/1098D6BA98D69E12/academics/SEMESTER_7/BTP/Bachelor_Thesis_Project/Exp1_HE/Train/'
trainydirname = '/media/rahul/1098D6BA98D69E12/academics/SEMESTER_7/BTP/Bachelor_Thesis_Project/Exp1_HE/Train_anno/'

traindataset = load_data(trainxdirname,trainydirname)
print traindataset[1].shape
print traindataset[1][0].shape
print traindataset[1][1].shape
print traindataset[1][2].shape