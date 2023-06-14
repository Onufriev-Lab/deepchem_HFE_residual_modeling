
import deepchem as dc
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib
import warnings
from utils import *
import timeit

warnings.filterwarnings('ignore')

freeSolve = pickle.load(open('dicts/consol.pickle', 'rb')) # FreeSolve Database

expPre, tip, smiles, dipole, gbn, igb, asc = [], [], [], [], [], [], []
for i in freeSolve.keys():
    #if('BGB+' not in freeSolve[i].keys()):
    #    continue
    expPre.append(freeSolve[i]['expt'])
    tip.append(freeSolve[i]['calc'])
    smiles.append(freeSolve[i]['smiles'])

n_arr = np.arange(50, 500, 50)##np.arange(10, len(smiles)//2, 5)
testr = []
trainr = []
t = timeit.default_timer()
for n in n_arr:
    print(str(n)+"  :  " + str(n_arr[-1]))
    testr.append([])
    trainr.append([])
    for i in range(20):
        #print(n, i)
        
        #smiles_s = subSample(smiles, s)
        #expPre_s = subSample(expPre, s)
        #tip_s = subSample(tip, s)

        b = psuedoScramble(expPre, bins=int(len(expPre)/10))
        ss = b[1::2]
        rs = b[0::2]

        #print("part")
        smilesTest, smilesTrain = partition(smiles, (ss, rs))
        expPreTest, expPreTrain = partition(expPre, (ss, rs))
        tipTest, tipTrain = partition(tip, (ss, rs))

        #s = np.random.permutation(n)
        #smilesTrain = subSample(smilesTrain, s)
        #expPreTrain = subSample(expPreTrain, s)
        #tipTrain = subSample(tipTrain, s)
        

        #print("feat")

        feat = dc.feat.ConvMolFeaturizer(per_atom_fragmentation=False)

        test = dc.data.NumpyDataset(
            X=feat.featurize(smilesTest), 
            y=np.array(expPreTest-tipTest).transpose())
        
        train = dc.data.NumpyDataset(
            X=feat.featurize(smilesTrain), 
            y=np.array(expPreTrain-tipTrain).transpose())


        #print("train")
        model = dc.models.GraphConvModel(n_tasks=1, mode='regression', dropout=0.2, batch_normalize=True, batch_size=100)
        model.fit(train, 500)
        
        #print("predict")
        pe1 = model.predict_on_batch(test.X).flatten()
        pe2 = model.predict_on_batch(train.X).flatten()

        testr[-1].append(rmsd(test.y-pe1))
        trainr[-1].append(rmsd(train.y-pe2))
        #print("done")

y = []
for i in testr:
    y.append(i)

plt.scatter(n_arr, y)
plt.savefig("ntest.png", dpi=50)

print("done in ", str(int((timeit.default_timer()-t)/1.0)) + " seconds")