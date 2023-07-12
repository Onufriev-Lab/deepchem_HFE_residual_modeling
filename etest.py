
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4' 

import deepchem as dc
import pandas as pd
import pickle
import numpy as np
import warnings

from utils import *
warnings.filterwarnings('ignore')

from matplotlib import pyplot as plt

def runExp(params, smiles, expt, feat, part):#partition (train, test, val)
    smiles = partition(smiles, part)
    expt = partition(expt, part)
    feat = partition(feat, part)

    featurizer = dc.feat.ConvMolFeaturizer(per_atom_fragmentation=False)
    
    train = dc.data.NumpyDataset(X=featurizer.featurize(smiles[0]), y=np.array(expt[0]-feat[0]).transpose())

    model = dc.models.GraphConvModel(n_tasks=1, graph_conv_layers=params['graph_conv_layers'], mode='regression', dropout=params['dropout'], batch_normalize=params['batch_normalize'], batch_size=params['batch_size'], dense_layer_size=params['dense_layer_size'])
    
    #x = []
    #for i in range(params['epochs']):
    ##    #model.fit(train, nb_epoch=1)
    #    x.append(model.fit(train, nb_epoch=1))
    #    print(x)
    #plt.plot(x, np.arange(len(x)))
    #plt.show()
    model.fit(train, nb_epoch=params['epochs'])

    p = ()

    for i in range(len(part)):
        p += (np.array(model.predict_on_batch(featurizer.featurize(smiles[i])).flatten()),)

    p_true = {'test' : list(expt[1]), 'train' : list(expt[0])}
    p_phy = {'test' : list(feat[1]), 'train' : list(feat[0])}
    p_corr = {'test' : list(p[1] + feat[1]), 'train' : list(p[0] + feat[0])}

    if(len(part) > 2):
        p_true['valid'] = list(expt[2])
        p_phy['valid'] = list(feat[2])
        p_corr['valid'] = list(p[2]+feat[2])

    return p_true, p_phy, p_corr



freeSolve = pickle.load(open('dicts/consol.pickle', 'rb')) # FreeSolve Database

expt, tip, smiles, gbn, igb, asc, zap, cha, nul = [], [], [], [], [], [], [], [], []
for i in freeSolve.keys():
    expt.append(freeSolve[i]['expt'])
    smiles.append(freeSolve[i]['smiles'])
    #
    tip.append(freeSolve[i]['calc'])
    gbn.append(freeSolve[i]['gbnsr6'])
    igb.append(freeSolve[i]['igb5'])
    asc.append(freeSolve[i]['asc'])
    zap.append(freeSolve[i]['zap9'])
    cha.append(freeSolve[i]['cha'])
    nul.append(0)
feats = {'tip3p' : tip, 'gbnsr6' : gbn, 'igb5' : igb, 'asc' : asc, 'null' : nul, 'zap9' : zap, 'cha' : cha}



def kfold(params, b = None, val = None):
    k = params['kfold']
    if(k == -1):
        return kfinal(params, b=b, val=val)

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
        
        if(type(val) != type(None)):
            part += (val,)
        #    print("(kth, train, test, val)  :  ", (i, len(part[0]), len(part[1]), len(part[2])))
        #else:
        #    print("(kth, train, test)  :  ", (i, len(part[0]), len(part[1])))
        
        
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

def kfinal(params, b = None, val = None):

    if(type(b) == type(None)):
        b = psuedoScramble(expt, bins=int(len(expt)/k))

    
    stats = []
    
    part = (b, val)
    p_true, p_phy, p_corr = runExp(params, smiles, expt, feats[params['feat']], part)

    
    stats.append({
        'phy_rmsd' :     {'test' : rmsd(        p_true['test'], p_phy ['test']), 'train' : rmsd(        p_true['train'], p_phy ['train'])},
        'ml_rmsd'  :     {'test' : rmsd(        p_true['test'], p_corr['test']), 'train' : rmsd(        p_true['train'], p_corr['train'])},
        'phy_md' :       {'test' : md(          p_true['test'], p_phy ['test']), 'train' : md(          p_true['train'], p_phy ['train'])},
        'ml_md'  :       {'test' : md(          p_true['test'], p_corr['test']), 'train' : md(          p_true['train'], p_corr['train'])},
        'phy_out_rmsd' : {'test' : ormsd(0.05,  p_true['test'], p_phy ['test']), 'train' : ormsd(0.05,  p_true['train'], p_phy ['train'])},
        'ml_out_rmsd'  : {'test' : ormsd(0.05,  p_true['test'], p_corr['test']), 'train' : ormsd(0.05,  p_true['train'], p_corr['train'])}
    })#

    return [p_true], [p_phy], [p_corr], stats




np.random.seed(10)
b = list(psuedoScramble(expt, bins=int(len(expt)/10)))
# params = [{'epochs' : 500, 'dropout' : 0.4, 'batch_normalize' : False, 'batch_size' : 100, 'feat' : 'tip3p', 'kfold' : -1, 'dense_layer_size' : 27, 'graph_conv_layers' : [32, 32]},
#          {'epochs' : 500, 'dropout' : 0.4, 'batch_normalize' : False, 'batch_size' : 100, 'feat' : 'gbnsr6', 'kfold' : -1, 'dense_layer_size' : 27, 'graph_conv_layers' : [32, 32]},
#          {'epochs' : 500, 'dropout' : 0.4, 'batch_normalize' : False, 'batch_size' : 100, 'feat' : 'igb5', 'kfold' : -1, 'dense_layer_size' : 27, 'graph_conv_layers' : [32, 32]},
#          {'epochs' : 500, 'dropout' : 0.4, 'batch_normalize' : False, 'batch_size' : 100, 'feat' : 'asc', 'kfold' : -1, 'dense_layer_size' : 27, 'graph_conv_layers' : [32, 32]}]
#
#params = [{'epochs' : 500, 'dropout' : 0.4, 'batch_normalize' : False, 'batch_size' : 100, 'feat' : 'null', 'kfold' : -1, 'dense_layer_size' : 27, 'graph_conv_layers' : [32, 32]}]
#params = [{'epochs' : 500, 'dropout' : 0.4, 'batch_normalize' : False, 'batch_size' : 100, 'feat' : 'zap9', 'kfold' : -1, 'dense_layer_size' : 27, 'graph_conv_layers' : [32, 32]},
#          {'epochs' : 500, 'dropout' : 0.4, 'batch_normalize' : False, 'batch_size' : 100, 'feat' : 'cha', 'kfold' : -1, 'dense_layer_size' : 27, 'graph_conv_layers' : [32, 32]},
#          {'epochs' : 500, 'dropout' : 0.4, 'batch_normalize' : False, 'batch_size' : 100, 'feat' : 'tip3p', 'kfold' : -1, 'dense_layer_size' : 27, 'graph_conv_layers' : [32, 32]},
#          {'epochs' : 500, 'dropout' : 0.4, 'batch_normalize' : False, 'batch_size' : 100, 'feat' : 'gbnsr6', 'kfold' : -1, 'dense_layer_size' : 27, 'graph_conv_layers' : [32, 32]},
#          {'epochs' : 500, 'dropout' : 0.4, 'batch_normalize' : False, 'batch_size' : 100, 'feat' : 'igb5', 'kfold' : -1, 'dense_layer_size' : 27, 'graph_conv_layers' : [32, 32]},
#          {'epochs' : 500, 'dropout' : 0.4, 'batch_normalize' : False, 'batch_size' : 100, 'feat' : 'asc', 'kfold' : -1, 'dense_layer_size' : 27, 'graph_conv_layers' : [32, 32]}]
params = [{'epochs' : 500, 'dropout' : 0.4, 'batch_normalize' : False, 'batch_size' : 100, 'feat' : 'tip3p', 
           'kfold' : -1, 'dense_layer_size' : 27, 'graph_conv_layers' : [32, 32]},
         {'epochs' : 500, 'dropout' : 0.4, 'batch_normalize' : False, 'batch_size' : 100, 'feat' : 'gbnsr6', 
          'kfold' : -1, 'dense_layer_size' : 27, 'graph_conv_layers' : [32, 32]},
         {'epochs' : 500, 'dropout' : 0.4, 'batch_normalize' : False, 'batch_size' : 100, 'feat' : 'asc', 
          'kfold' : -1, 'dense_layer_size' : 27, 'graph_conv_layers' : [32, 32]},
         {'epochs' : 500, 'dropout' : 0.4, 'batch_normalize' : False, 'batch_size' : 100, 'feat' : 'igb5', 
          'kfold' : -1, 'dense_layer_size' : 27, 'graph_conv_layers' : [32, 32]},
         {'epochs' : 500, 'dropout' : 0.4, 'batch_normalize' : False, 'batch_size' : 100, 'feat' : 'zap9', 
          'kfold' : -1, 'dense_layer_size' : 27, 'graph_conv_layers' : [32, 32]},          
         {'epochs' : 500, 'dropout' : 0.4, 'batch_normalize' : False, 'batch_size' : 100, 'feat' : 'cha', 
          'kfold' : -1, 'dense_layer_size' : 27, 'graph_conv_layers' : [32, 32]},
         {'epochs' : 500, 'dropout' : 0.4, 'batch_normalize' : False, 'batch_size' : 100, 'feat' : 'null', 
          'kfold' : -1, 'dense_layer_size' : 27, 'graph_conv_layers' : [32, 32]}]
val = []
for i in range(len(b)//8):
    #j = np.random.randint(0, len(b))
    j = i*7
    val.append(b.pop(j))

for p in params:
    np.random.seed(10)
    # print(p)
    print(p['feat'])
    p_true, p_phy, p_corr, stats = [], [], [], []
    for i in range(1):
        pt, pp, pc, ps = kfold(p, b, val)
        p_true += pt
        p_phy += pp
        p_corr += pc
        stats = ps

    pt, pp, pc = {'test':[], 'train':[], 'valid':[]}, {'test':[], 'train':[], 'valid':[]}, {'test':[], 'train':[], 'valid':[]}
    for i in range(len(p_true)):
        for j in p_true[i]:
            pt[j] += p_true[i][j]
            pp[j] += p_phy[i][j]
            pc[j] += p_corr[i][j]

    d = {'p_true' : pt, 'p_phy' : pp, 'p_corr' : pc}

    s = {}
    #    'phy_rmsd' :     {'test' : [], 'train' : [], 'valid' : []},
    #    'ml_rmsd'  :     {'test' : [], 'train' : [], 'valid' : []},
    #    'phy_md' :       {'test' : [], 'train' : [], 'valid' : []},
    #    'ml_md'  :       {'test' : [], 'train' : [], 'valid' : []},
    #    'phy_out_rmsd' : {'test' : [], 'train' : [], 'valid' : []},
    #    'ml_out_rmsd'  : {'test' : [], 'train' : [], 'valid' : []}
    #}
    
    print('physics model: test',stats[0]['phy_rmsd']['test'],'train',stats[0]['phy_rmsd']['train'])
    print('physics + ml: test',stats[0]['ml_rmsd']['test'],'train',stats[0]['ml_rmsd']['train'])
    print()
# for p in params:
#     np.random.seed(10)
#     print(p)
#     p_true, p_phy, p_corr, stats = [], [], [], []
#     for i in range(100):
#         print(i)
#         pt, pp, pc, ps = kfold(p, b, val)
#         p_true += pt
#         p_phy += pp
#         p_corr += pc
#         stats += ps

#     pt, pp, pc = {'test':[], 'train':[], 'valid':[]}, {'test':[], 'train':[], 'valid':[]}, {'test':[], 'train':[], 'valid':[]}
#     for i in range(len(p_true)):
#         for j in p_true[i]:
#             pt[j] += p_true[i][j]
#             pp[j] += p_phy[i][j]
#             pc[j] += p_corr[i][j]

#     d = {'p_true' : pt, 'p_phy' : pp, 'p_corr' : pc}

#     s = {}
#     #    'phy_rmsd' :     {'test' : [], 'train' : [], 'valid' : []},
#     #    'ml_rmsd'  :     {'test' : [], 'train' : [], 'valid' : []},
#     #    'phy_md' :       {'test' : [], 'train' : [], 'valid' : []},
#     #    'ml_md'  :       {'test' : [], 'train' : [], 'valid' : []},
#     #    'phy_out_rmsd' : {'test' : [], 'train' : [], 'valid' : []},
#     #    'ml_out_rmsd'  : {'test' : [], 'train' : [], 'valid' : []}
#     #}

#     for i in stats[0]:
#         s[i] = {}
#         for j in stats[0][i]:
#             s[i][j] = []

#     for i in stats:
#         for j in i:
#             for k in i[j]:
#                 s[j][k].append(i[j][k])

#     #for j in s:
#     #    for k in s[j]:
#     #        s[j][k] = np.mean(s[j][k])

    # pickle.dump(s, open('runs/final_' + p['feat'] + '_dropout_' + str(p['dropout']) + '_dense_' + str(p['dense_layer_size']), 'wb'))
    # pickle.dump(d, open('runs/final_' + p['feat'] + '_dropout_' + str(p['dropout']) + '_dense_' + str(p['dense_layer_size']) + '_points', 'wb'))