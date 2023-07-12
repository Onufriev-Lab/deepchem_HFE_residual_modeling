
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4' 

import deepchem as dc
import pandas as pd
import pickle
import numpy as np
import warnings

from utils import *
warnings.filterwarnings('ignore')

def runExp(params, smiles, expt, feat, part):#partition (train, test, val)
    smiles = partition(smiles, part)
    expt = partition(expt, part)
    feat = partition(feat, part)

    featurizer = dc.feat.ConvMolFeaturizer(per_atom_fragmentation=False)
    
    train = dc.data.NumpyDataset(X=featurizer.featurize(smiles[0]), y=np.array(expt[0]-feat[0]).transpose())

    model = dc.models.GraphConvModel(n_tasks=1, graph_conv_layers=params['graph_conv_layers'], mode='regression', dropout=params['dropout'], batch_normalize=params['batch_normalize'], batch_size=params['batch_size'], dense_layer_size=params['dense_layer_size'])
    model.fit(train, nb_epoch=params['epochs'])

    p = ()
    for i in range(len(part)):
        p += (np.array(model.predict_on_batch(featurizer.featurize(smiles[i])).flatten()),)

    p_true = {'test' : list(expt[1]), 'train' : list(expt[0])}
    p_phy = {'test' : list(feat[1]), 'train' : list(feat[0])}
    p_corr = {'test' : list(p[1]) + feat[1], 'train' : list(p[0] + feat[0])}

    if(len(part) > 2):
        p_true['valid'] = list(expt[2])
        p_phy['valid'] = list(feat[2])
        p_corr['valid'] = list(p[2]+feat[2])

    return p_true, p_phy, p_corr



freeSolve = pickle.load(open('dicts/consol.pickle', 'rb')) # FreeSolve Database

expt, tip, smiles, gbn, igb, asc, nul = [], [], [], [], [], [], []
for i in freeSolve.keys():
    expt.append(freeSolve[i]['expt'])
    smiles.append(freeSolve[i]['smiles'])
    #
    tip.append(freeSolve[i]['calc'])
    gbn.append(freeSolve[i]['gbnsr6'])
    igb.append(freeSolve[i]['igb5'])
    asc.append(freeSolve[i]['asc'])
    nul.append(0)
    feats = {'tip3p' : tip, 'gbnsr6' : gbn, 'igb5' : igb, 'asc' : asc, 'null' : nul}



def kfold(params, b = None, val = None):
    k = params['kfold']
    if(type(b) == type(None)):
        b = psuedoScramble(expt, bins=int(len(expt)/k))

    folds = []
    for i in range(k):
        folds.append(b[i::k])
    
    p_true = []
    p_phy = []
    p_corr = []
    stats = []
    for i in range(k):
        part = (np.hstack(tuple(folds[:i])+tuple(folds[(i+1):])), folds[i])#train, test, val
        
        #if(type(val) != type(None)):
        #    part += (val,)
        #    print("(kth, train, test, val)  :  ", (i, len(part[0]), len(part[1]), len(part[2])))
        #else:
        #    print("(kth, train, test)  :  ", (i, len(part[0]), len(part[1])))
        #print(" " + str(i)),

        pt, pp, pc = runExp(params, smiles, expt, feats[params['feat']], part)

        p_true.append(pt)
        p_phy.append(pp)
        p_corr.append(pc)
       

        if(type(val) != type(None)):
             stats.append({
                'phy_rmsd' :     {'test' : rmsd(        pt['test'], pp['test']), 'train' : rmsd(        pt['train'], pp['train']), 'valid' : rmsd(       pt['valid'], pp['valid'])},
                'ml_rmsd'  :     {'test' : rmsd(        pt['test'], pc['test']), 'train' : rmsd(        pt['train'], pc['train']), 'valid' : rmsd(       pt['valid'], pc['valid'])},
                'phy_md' :       {'test' : md(          pt['test'], pp['test']), 'train' : md(          pt['train'], pp['train']), 'valid' : md(         pt['valid'], pp['valid'])},
                'ml_md'  :       {'test' : md(          pt['test'], pc['test']), 'train' : md(          pt['train'], pc['train']), 'valid' : md(         pt['valid'], pc['valid'])},
                'phy_out_rmsd' : {'test' : ormsd(0.05,  pt['test'], pp['test']), 'train' : ormsd(0.05,  pt['train'], pp['train']), 'valid' : ormsd(0.05, pt['valid'], pp['valid'])},
                'ml_out_rmsd'  : {'test' : ormsd(0.05,  pt['test'], pc['test']), 'train' : ormsd(0.05,  pt['train'], pc['train']), 'valid' : ormsd(0.05, pt['valid'], pc['valid'])}
            })#
        else:
             stats.append({
                'phy_rmsd' :     {'test' : rmsd(        pt['test'], pp['test']), 'train' : rmsd(        pt['train'], pp['train'])},
                'ml_rmsd'  :     {'test' : rmsd(        pt['test'], pc['test']), 'train' : rmsd(        pt['train'], pc['train'])},
                'phy_md' :       {'test' : md(          pt['test'], pp['test']), 'train' : md(          pt['train'], pp['train'])},
                'ml_md'  :       {'test' : md(          pt['test'], pc['test']), 'train' : md(          pt['train'], pc['train'])},
                'phy_out_rmsd' : {'test' : ormsd(0.05,  pt['test'], pp['test']), 'train' : ormsd(0.05,  pt['train'], pp['train'])},
                'ml_out_rmsd'  : {'test' : ormsd(0.05,  pt['test'], pc['test']), 'train' : ormsd(0.05,  pt['train'], pc['train'])}
            })#

    return p_true, p_phy, p_corr, stats
    

b = list(psuedoScramble(expt, bins=int(len(expt)/10)))

'''val = []
for i in range(len(b)//2):
    j = np.random.randint(0, len(b))
    val.append(b.pop(j))

p_true, p_phy, p_corr, stats = kfold(params, b, val)
'''

dl_arr = np.array(10*np.power(1.65, np.arange(0, 10, 1)), int)
drop_arr = np.arange(0, 1, 0.1)
cl_arr = np.arange(120, 180, 4)
#for dl in dl_arr:
#    for drop in drop_arr:
        

#for dl in dl_arr:
#    for drop in drop_arr:
for cl in cl_arr:
    #params = {'graph_conv_layers' : [64, 64], 'epochs' : 500, 'dropout' : drop, 'batch_normalize' : False, 'batch_size' : 100, 'feat' : 'null', 'kfold' : 10, 'dense_layer_size' : dl}
    params = {'graph_conv_layers' : [cl, cl], 'epochs' : 500, 'dropout' : 0.6, 'batch_normalize' : False,'batch_size'  : 100, 'feat' : 'tip3p', 'kfold' : 10, 'dense_layer_size' : 27}
    #
    p_true = []
    p_phy = []
    p_corr = []
    stats = []
    print(cl)
    #print(dl, ", ", drop)
    for i in range(3):
        #print((cl, i))
        pt, pp, pc, s = kfold(params)
        p_true += pt
        p_phy += pp
        p_corr += pc
        stats += s
    d = {'true' : p_true, 'phy' : p_phy, 'corr' : p_corr, 'stats' : stats, 'params' : params}
    print('physics model: test',stats[0]['phy_rmsd']['test'],'train',stats[0]['phy_rmsd']['train'])
    print('physics + ml: test',stats[0]['ml_rmsd']['test'],'train',stats[0]['ml_rmsd']['train'])
    print()
    #pickle.dump(d, open('HyperExpNull2/' + 'dl_' + str(dl)[:5] + '_dr_' + str(drop)[:5], 'wb'))
    pickle.dump(d, open('HyperExpNull2/' + 'cl_' + str(cl)[:5], 'wb'))
    #pickle.dump(d, open('HyperExp/' + 'dl_' + str(dl) + '_dr_' + str(drop), 'wb'))

