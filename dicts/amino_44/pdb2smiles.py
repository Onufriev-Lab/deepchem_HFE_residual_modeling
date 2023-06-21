
from rdkit import Chem
import os
import pandas as pd
import pickle

if('pdb' not in os.listdir()):
    print('could not find pdb folder')
    quit()

tip_file = pd.read_csv('TI_amino_48.dat', sep=' ', header=None).to_numpy()
tip_dict = {}
for i in tip_file:
    tip_dict[i[0]] = {}
    tip_dict[i[0]]['tip3p'] = i[1]


for f in os.listdir('pdb'):
    mol = Chem.rdmolfiles.MolFromPDBFile('pdb/'+f)
    n = f.replace('.pdb', '')
    smiles = Chem.rdmolfiles.MolToSmiles(mol)
    if(n not in tip_dict.keys()):
        print("missing tip3p value for " + n)
        continue
    print((' '+n+" ")[-9:], (str(tip_dict[n]['tip3p'])+'    ')[:8], smiles)
    tip_dict[n]['smiles'] = smiles


pickle.dump(tip_dict, open("amino_smiles.pickle", 'wb'))


