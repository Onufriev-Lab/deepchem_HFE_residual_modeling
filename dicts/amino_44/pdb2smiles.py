
from rdkit import Chem
import os
import pandas as pd

if('pdb' not in os.listdir()):
    print('could not find pdb folder')
    quit()

tip_file = pd.read_csv('TI_amino_48.dat', sep=' ', header=None).to_numpy()
tip_dict = {}
for i in tip_file:
    tip_dict[i[0]] = i[1]

os.chdir('pdb')
for f in os.listdir():
    mol = Chem.rdmolfiles.MolFromPDBFile(f)
    n = f.replace('.pdb', '')
    if(n not in tip_dict.keys()):
        print("missing tip3p value for " + n)
        continue
    print(n, tip_dict[n], Chem.rdmolfiles.MolToSmiles(mol))



