
import deepchem as dc
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

freeSolve = pickle.load(open('./FreeSolv/database.pickle', 'rb')) # FreeSolve Database
convName = pd.read_csv('full_correspondence', sep=';', header=None).to_numpy()[:, 0] # convert from FreeSolve to dans bs
convID = pd.read_csv('full_correspondence', sep=';', header=None).to_numpy()[:, 2]
conv = {convName[i] : convID[i] for i in range(len(convName))}

#danName = np.hstack((pd.read_csv('teste.csv')['COMPONENT'].to_numpy()[1:], pd.read_csv('traine.csv')['COMPONENT'].to_numpy()[1:]))# physical model prediction for test and train 
#phyPre = np.hstack((pd.read_csv('teste.csv')['POLAR+NONPOLAR'].to_numpy()[1:], pd.read_csv('traine.csv')['POLAR+NONPOLAR'].to_numpy()[1:]))

#id, experimental, smiles
id, expPre, phyPre, smiles = [], [], [], []

"""for i in range(len(danName)):#rigid subset
    id.append(conv[danName[i]])
    expPre.append(freeSolve[id[i]]['expt'])
    smiles.append(freeSolve[id[i]]['smiles'])"""

for i in freeSolve.keys():#entire freesolve
    id.append(i)
    smiles.append(freeSolve[i]['smiles'])
    expPre.append(freeSolve[i]['expt'])
    phyPre.append(freeSolve[i]['calc'])

id, expPre, phyPre, smiles = np.array(id), np.array(expPre), np.array(phyPre), np.array(smiles)

X = dc.feat.ConvMolFeaturizer(per_atom_fragmentation=False).featurize(smiles)
y = np.array(expPre-phyPre).transpose()
#y = np.array(expPre).transpose()



dataset = dc.data.NumpyDataset(X=X, y=y)

sx = []# test plot points
sy = []
srms = []# test rms

rx = []# train plot points
ry = []
rrms = []# train rms

for i in range(10):
    splitter = dc.splits.RandomSplitter()
    train, test = splitter.train_test_split(dataset)
    #
    model = dc.models.GraphConvModel(n_tasks=1, mode='regression', dropout=0.2)
    model.fit(train, nb_epoch=1000)
    #
    pe1 = model.predict_on_batch(test.X).flatten()
    pe2 = model.predict_on_batch(train.X).flatten()
    #
    sx = np.hstack((sx, test.y))
    sy = np.hstack((sy, pe1))
    #
    rx = np.hstack((rx, train.y))
    ry = np.hstack((ry, pe2))
    #
    srms.append(np.sqrt(sum(np.power(test.y-pe1, 2))/(test.y.size)))
    rrms.append(np.sqrt(sum(np.power(train.y-pe2, 2))/(train.y.size)))
    print(i)


#print("Train error: ", np.sqrt(sum(abs(train.y-pe2))/(train.y.size)))
#print("Test  error: ", np.sqrt(sum(abs(test.y-pe1))/(test.y.size)))

f = (min(min(rx), min(sx), min(ry), min(sy)), max(max(rx), max(sx), max(ry), max(sy)))

plt.scatter(rx, ry, c='red', label='train', s = 1)
plt.scatter(sx, sy, c='green', label='test', s = 2)
plt.xlabel("Physical Model Error kcal/mol")
plt.ylabel("Predicted Model Error kcal/mol")
plt.title("1000 epochs")
plt.plot([f[0], f[1]], [f[0], f[1]], linestyle='dotted', c='black')
plt.legend()

plt.text(-0, f[0]+0.2*(f[1]-f[0]), "Train RMSD: "+ str(np.mean(rrms))[:5] + " +- " + str(np.std(rrms))[:5])
plt.text(-0, f[0]+0.1*(f[1]-f[0]), "Test  RMSD: "+ str(np.mean(srms))[:5] + " +- " + str(np.std(srms))[:5])
plt.savefig('MLRigidSolvation.png', dpi=1000)
plt.show()

