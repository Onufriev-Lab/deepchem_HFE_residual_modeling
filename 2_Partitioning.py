

from utils import *
from matplotlib import pyplot as plt

import_from('1_Dictgen', ['pickle', 'np', 'load_data'], __name__)

def subSample(a, i):# index sampling
    c = []
    for j in i:
        c.append(a[j])
    return np.array(c)

def psuedoScramble(v, a = None, bins=None):# evenly splits a based on v
    if(type(bins) == type(None)):
        bins = int(len(v)/20)
    if(type(a) == type(None)):
        a = np.arange(len(v))
    bin=[]
    v = np.argsort(v)
    binw = len(v)/float(bins)
    for i in range(bins):
        v[int(binw*i):int(binw*i+binw)] = np.random.permutation(v[int(binw*i):int(binw*i+binw)])
    return subSample(a, v)

def partition(idx, *data): # given an array of index groups return an array of partitions
    a = ()
    for dat in data:
        b = {}
        for i in idx:
            b[i] = subSample(dat, idx[i])
        a += (b,)
    return a





if __name__ == '__main__':
    feats = load_data()
    k = 20
    '''np.random.seed(10)
    b = list(psuedoScramble(feats['expt'], bins=int(len(feats['expt'])/k)))
    val = []
    folds = []
    for i in range(len(b)//8):
        val.append(b.pop(i*7))
    for i in range(k):
        folds.append(b[i:k])'''

    e = feats['expt']
    a = np.arange(len(e))
    plt.bar(np.argsort(a), e)
    plt.savefig('tmp.png', dpi=200)
    print('pre-sort')
    plt.clf()
    input()

    a = np.argsort(e)
    plt.bar(np.argsort(a), e)
    plt.savefig('tmp.png', dpi=200)
    print('sorted')
    plt.clf()
    input()

    b = list(psuedoScramble(feats['expt'], bins=int(len(feats['expt'])/k)))
    plt.bar(np.argsort(b), e)
    plt.savefig('tmp.png', dpi=200)
    print('scrambeled')
    plt.clf()
    input()