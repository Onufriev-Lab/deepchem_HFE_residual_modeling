
import pickle
import pandas as pd
import numpy as np

freesolv = pickle.load(open('database.pickle', 'rb'))

full_correspondence = pd.read_csv('BGB+/full_correspondence', header=None, sep=';').to_numpy()


rigid_names = full_correspondence[:,0]
rigid_ids = full_correspondence[:,2]
name_to_id = {}
for i in np.vstack((rigid_names, rigid_ids)).T:
    name_to_id[i[0]] = i[1]


rigid_test_names = pd.read_csv('BGB+/teste.csv')['COMPONENT'].to_numpy()[1:]
rigid_test_BGB = pd.read_csv('BGB+/teste.csv')['POLAR+NONPOLAR'].to_numpy()[1:]

for i in np.vstack((rigid_test_names, rigid_test_BGB)).T:
    if('BGB+' in freesolv[name_to_id[i[0]]].keys()):
        print(i, name_to_id[i[0]])
    freesolv[name_to_id[i[0]]]['BGB+'] = i[1]
    freesolv[name_to_id[i[0]]]['BGB+group'] = 'test'

rigid_train_names = pd.read_csv('BGB+/traine.csv')['COMPONENT'].to_numpy()[1:]
rigid_train_BGB = pd.read_csv('BGB+/traine.csv')['POLAR+NONPOLAR'].to_numpy()[1:]

for i in np.vstack((rigid_train_names, rigid_train_BGB)).T:
    if('BGB+' in freesolv[name_to_id[i[0]]].keys()):
        print(i, name_to_id[i[0]])
    freesolv[name_to_id[i[0]]]['BGB+'] = i[1]
    freesolv[name_to_id[i[0]]]['BGB+group'] = 'train'


names = pd.read_csv('asc-igb/names', header=None).to_numpy().flatten()

asc_nse = pd.read_csv('asc-igb/ascnse', header=None).to_numpy().flatten()
asc_pse = pd.read_csv('asc-igb/ascpse', header=None).to_numpy().flatten()

igb_nse = pd.read_csv('asc-igb/igb5nse', header=None).to_numpy().flatten()
igb_pse = pd.read_csv('asc-igb/igb5pse', header=None).to_numpy().flatten()

for i in np.vstack((names, asc_nse+asc_pse, igb_nse+igb_pse)).T:
    freesolv[i[0]]['asc'] = i[1]
    freesolv[i[0]]['igb5'] = i[2]

gbn = pd.read_csv('GBNSR6_zap9.csv', header=None, sep=' ').to_numpy()
for i in gbn:
    freesolv[i[0][:-4]]['gbnsr6'] = i[1]

zap = pd.read_csv("Molname_ZAP9_EXP", header=None, sep='   ', engine='python').to_numpy()
for i in zap:
    freesolv[i[0]]['zap9'] = i[1]
    if(np.abs(freesolv[i[0]]['expt'] - i[2])>0.01):
        print("mismatch in zap: ", i[0])

cha = pd.read_csv("Molname_CHAGB_EXP", header=None, sep='   ', engine='python').to_numpy()
for i in cha:
    freesolv[i[0]]['cha'] = i[1]
    if(np.abs(freesolv[i[0]]['expt'] - i[2])>0.01):
        print("mismatch in cha: ", i[0])

bestgb = pd.read_csv("FreeSolv_GBNSR6_ene_642_bestgb.dat", header=None, sep=' ', engine='python').to_numpy()
for i in range(bestgb.shape[0]):
    freesolv[cha[i,0]]['bestgb'] = bestgb[i, 2]
        
pickle.dump(freesolv, open('consol.pickle', 'wb'))