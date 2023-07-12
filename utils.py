
import numpy as np
import sys

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
import warnings
warnings.filterwarnings('ignore')


def lerp(x1, x2, k): # linear interpolation
    return x1+(x2-x1)*k

def ilerp(x1, x2, k): # inverse linear interpolation
    return (k-x1)/(x2-x1)

def mapRange(x1, x2, y1, y2, k): # linear mapping between two intervals
    return lerp(y1, y2, ilerp(x1, x2, k))

def rmsd(a, b=None): # root mean square deviation
    if(type(b) == type(None)):
        b = np.zeros(len(a))
    return np.sqrt(np.mean(np.power(np.array(a)-np.array(b), 2)))

def md(a, b=None): # mean devialtion
    if(type(b) == type(None)):
        b = np.zeros(len(a))
    return np.mean(np.array(a)-np.array(b))

def ormsd(p, a, b=None): # rnsd of p fraction of outliers
    if(type(b) == type(None)):
        b = np.zeros(len(a))
    o = np.sort(np.abs(np.array(a)-np.array(b)))[int((1.0-p)*len(a)):]
    return rmsd(o)



def hist2D(x, y, bins=10):
    a = np.zeros((bins, bins))
    xp = np.array(mapRange(min(x), max(x)+0.0001, 0, bins, x), int)
    yp = np.array(mapRange(min(y), max(y)+0.0001, 0, bins, y), int)
    for i in np.vstack((xp, yp)).T:
        a[i[0], i[1]] += 1
    return a



def import_from(module, names, location):
    module = __import__(module, fromlist=names)
    for name in names:
        #globals()[name] = getattr(module, name)
        #print(globals().keys())
        setattr(sys.modules[location], name, getattr(module, name))


def serialize(m):

    if(hasattr(m, 'weights')):
        for i in serialize(m.weights):
            yield '.weights'+i
    elif(hasattr(m, '__iter__') and not (hasattr(m, 'shape') and len(m.shape)==0)):
        k = 0
        for j in m:
            for i in serialize(j):
                yield '['+str(k)+']'+i
            k+=1
    elif(hasattr(m, 'numpy')):
       yield ''#'.numpy()'
    else:
        print('failed at: ', m)
        yield ''

def soft_mkdir(name):
    try:
        os.mkdir(name)
    except FileExistsError:
        print(end='')

