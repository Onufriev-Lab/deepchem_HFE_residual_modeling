

import deepchem as dc

from utils import *

import_from('2_Partitioning', ['pickle', 'np', 'psuedoScramble', 'partition', 'load_data'], __name__)

def runExp(params, smiles, expt, feat):#partition (train, test, val)

    featurizer = dc.feat.ConvMolFeaturizer(per_atom_fragmentation=False)
    train = dc.data.NumpyDataset(X=featurizer.featurize(smiles['train']), y=np.subtract(expt['train'], feat['train']).transpose())

    model = dc.models.GraphConvModel(n_tasks=1, graph_conv_layers=params['graph_conv_layers'], mode='regression', dropout=params['dropout'], batch_normalize=params['batch_normalize'], batch_size=params['batch_size'], dense_layer_size=params['dense_layer_size'], model_dir=params['model_dir'])
    model.fit(train, nb_epoch=params['epochs'], log=False)

    p_true, p_phy, p_corr = {}, {}, {} # individual point data
    for i in smiles:
        p_true[i] = list(expt[i])
        p_phy[i]  = list(feat[i])
        p_corr[i] = list(np.array(feat[i])+np.array(model.predict_on_batch(featurizer.featurize(smiles[i])).flatten()))

    return p_true, p_phy, p_corr, model




if __name__ == '__main__':
    params = {'epochs' : 500, 'dropout' : 0.4, 'batch_normalize' : False, 'batch_size' : 1000, 'feat' : 'tip3p', 'kfold' : -1, 'dense_layer_size' : 27, 'graph_conv_layers' : [32, 32], 'model_dir' : 'test.model'}

    feats = load_data()

    np.random.seed(10)
    b = list(psuedoScramble(feats['expt']))
    parts = {}
    parts['test'] = []
    for i in range(len(b)//8):
        parts['test'].append(b.pop(i*7))
    parts['train'] = b

    smiles, expt, feat = partition(parts, feats['smiles'], feats['expt'], feats['tip3p'])
    pt, pp, pc, m = runExp(params, smiles, expt, feat)
    for i in smiles:
        print(i, end=': ')
        print(format(rmsd(pt[i], pp[i]), '.2f') + ' -> ' + format(rmsd(pt[i], pc[i]), '.2f'))


