
import numpy as np

empty_feats = [ 5,  9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
       26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
       43, 49, 50, 51, 52, 53, 54, 60, 61, 62, 63, 67, 68]

def lerp(x1, x2, k):
    return x1+(x2-x1)*k

def ilerp(x1, x2, k):
    return (k-x1)/(x2-x1)

def mapRange(x1, x2, y1, y2, k):
    return lerp(y1, y2, ilerp(x1, x2, k))

def rmsd(a, b=None):
    if(type(b) == type(None)):
        b = np.zeros(len(a))
    return np.sqrt(np.mean(np.power(np.array(a)-np.array(b), 2)))

def md(a, b=None):
    if(type(b) == type(None)):
        b = np.zeros(len(a))
    return np.mean(np.array(a)-np.array(b))

def ormsd(p, a, b=None):
    if(type(b) == type(None)):
        b = np.zeros(len(a))
    o = np.sort(np.abs(np.array(a)-np.array(b)))[int((1.0-p)*len(a)):]
    return rmsd(o)

def subSample(a, i):# index sampling
    c = []
    for j in i:
        c.append(a[j])
    return np.array(c)

def psuedoScramble(v, a = None, bins=10):# evenly splits a based on v
    if(a == None):
        a = np.arange(len(v))
    bin=[]
    v = np.argsort(v)
    binw = len(v)/float(bins)
    for i in range(bins):
        v[int(binw*i):int(binw*i+binw)] = np.random.permutation(v[int(binw*i):int(binw*i+binw)])
    return subSample(a, v)

def partition(data, idx):
    a = []
    for i in range(len(idx)):
        a.append(subSample(data, idx[i]))
    return tuple(a)

def addAtomFeat(atom_features, feat_num, feat):#adds feature to single atom
    #print(feat.shape)
    atom_features[:,empty_feats[feat_num]] = feat
    return atom_features

def addMolFeat(data, feat_num, feat):#adds per molecule feature
    for i in range(len(data)):
        data[i].atom_features = addAtomFeat(data[i].atom_features, feat_num, feat[i]*np.ones(data[i].atom_features.shape[0]))

def hist2D(x, y, bins=10):
    a = np.zeros((bins, bins))
    xp = np.array(mapRange(min(x), max(x)+0.0001, 0, bins, x), int)
    yp = np.array(mapRange(min(y), max(y)+0.0001, 0, bins, y), int)
    for i in np.vstack((xp, yp)).T:
        a[i[0], i[1]] += 1
    return a




def show(d, feat):
    print('               train            test', end='')
    if('valid' in d[list(d.keys())[0]].keys()):
        print('           test')
    else:
        print()
    
    feat = (feat + "        ")[:5]

    for i in d:
        k = i
        if(i=='phy_rmsd'):
            k = feat + ' rmsd    '
        elif(i=='ml_rmsd'):
            k = 'ml    rmsd    '
        elif(i=='phy_md'):
            k = feat + ' md      '
        elif(i=='ml_md'):
            k = 'ml    md      '
        elif(i=='phy_out_rmsd'):
            k = feat + ' outlier '
        elif(i=='ml_out_rmsd'):
            k = 'ml    outlier '
        elif(i=='null->phy'):
            k = 'null->phy'
        elif(i=='phy->ml'):
            k = 'phy->ml'
        elif(i=='null->ml'):
            k = 'null->ml'
        else:
            continue
        print(k, end='')
        for j in range(18-len(str(k))):
            print(' ', end='')
        print(format(np.mean(d[i]['train']), '.2f') + '+-' + format(np.std(d[i]['train']), '.2f'), '\t', end='')
        print(format(np.mean(d[i]['test' ]), '.2f') + '+-' + format(np.std(d[i]['test' ]), '.2f'), '\t', end='')

        if('valid' in d[i].keys()):
            print(format(np.mean(d[i]['valid']), '.2f') + '+-' + format(np.std(d[i]['valid']), '.2f'), '\t', end='')
        print()
    

