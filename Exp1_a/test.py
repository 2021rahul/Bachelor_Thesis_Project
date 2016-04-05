import os
#os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=gpu,floatX=float32'
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import random
from PIL import Image
import scipy.io

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
    
def load_tl_img(x,img):
    yval = np.zeros((1,1))
    nrow = img[0].shape[0]
    ncol = img[0].shape[1]
    xgen = []
    ygen = []    
    
    for y in range(25,390+25):
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
    
    for y in range(ncol-25-390,ncol-25):
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
    
    
def load_br_img(x,img):
    yval = np.zeros((1,1))
    nrow = img[0].shape[0]
    ncol = img[0].shape[1]
    xgen = []
    ygen = []    
    
    for y in range(ncol-25-390,ncol-25):
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
    
    
trainxdirname = '/media/rahul/1098D6BA98D69E12/academics/SEMESTER_7/BTP/Bachelor_Thesis_Project/Exp1_a/Test/'
trainydirname = '/media/rahul/1098D6BA98D69E12/academics/SEMESTER_7/BTP/Bachelor_Thesis_Project/Exp1_a/Test_anno/'

n_binmap_batches = 390

testdataset = load_data(trainxdirname,trainydirname)

for im in range(0,80):
    nrow = testdataset[im][0].shape[0]
    for i in xrange(n_binmap_batches):
        binmap_x_tl,binmap_y_tl,ixgen_tl,iygen_tl = load_tl_img(i+25,testdataset[im])
        binmap_x_tr,binmap_y_tr,ixgen_tr,iygen_tr = load_tr_img(i+25,testdataset[im])
        binmap_x_bl,binmap_y_bl,ixgen_bl,iygen_bl = load_bl_img(nrow-i-26,testdataset[im])
        binmap_x_br,binmap_y_br,ixgen_br,iygen_br = load_br_img(nrow-i-26,testdataset[im])

