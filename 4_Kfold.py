
from utils import *
import_from('3_ErrorPrediction', ['pickle', 'np', 'psuedoScramble', 'partition', 'load_data', 'dc', 'runExp'], __name__)

class Kfold:
    def __init__(self, data, k, b = None):
        if(type(b) == type(None)):
            self.b = list(psuedoScramble(data, bins=int(len(data)/k)))
        else:
            self.b = b
            
        self.test = []
        for i in range(len(self.b)//8):
            self.test.append(self.b.pop(i*7))
        self.folds = []

        if(k <= 0):
            k = 1
        for i in range(k):
            self.folds.append(self.b[i::k])
    
    def kfolds(self, *args, final = False):
        if(final or len(self.folds) == 1):
            parts = {'train' : np.hstack(self.folds), 'test' : self.test}
            yield partition(parts, *args)
        else:    
            for i in range(len(self.folds)):
                parts = {'train' : np.hstack(tuple(self.folds[:i])+tuple(self.folds[(i+1):])), 'valid' : self.folds[i], 'test' : self.test}
                yield partition(parts, *args)
                

def kfold(params, feats, b = None, val = None, final = False, folds = None):

    if(type(folds) == type(None)):
        folds = Kfold(feats['expt'], params['kfold'], b=b)

    p_true = []
    p_phy = []
    p_corr = []
    stats = []
    for k_smiles, k_expt, k_feat in folds.kfolds(feats['smiles'], feats['expt'], feats[params['feat']], final=final):
        
        
        pt, pp, pc, _ = runExp(params, k_smiles, k_expt, k_feat)

        for i in k_smiles:
            print(i, end=': ')
            print(format(rmsd(pt[i], pp[i]), '.2f') + ' -> ' + format(rmsd(pt[i], pc[i]), '.2f'))

        p_true.append(pt)
        p_phy.append(pp)
        p_corr.append(pc)

        s = {'phy_rmsd' : {}, 'ml_rmsd' : {}, 'phy_md' : {}, 'ml_md' : {}, 'phy_out_rmsd' : {}, 'ml_out_rmsd' : {}}
        for i in pt:
            s['phy_rmsd'    ][i] = rmsd(        pt[i], pp[i])
            s['ml_rmsd'     ][i] = rmsd(        pt[i], pc[i])
            s['phy_md'      ][i] = md(          pt[i], pp[i])
            s['ml_md'       ][i] = md(          pt[i], pc[i])
            s['phy_out_rmsd'][i] = ormsd(0.05,  pt[i], pp[i])
            s['ml_out_rmsd' ][i] = ormsd(0.05,  pt[i], pc[i])
        s['params'] = params
        stats.append(s)

    return p_true, p_phy, p_corr, stats


if __name__ == '__main__':
    params = {'epochs' : 500, 'dropout' : 0.4, 'batch_normalize' : False, 'batch_size' : 1000, 'feat' : 'tip3p', 'kfold' : 20, 'dense_layer_size' : 27, 'graph_conv_layers' : [32, 32], 'model_dir' : 'test.model', 'model_dir' : None}
    feats = load_data()

    np.random.seed(10)
    b = list(psuedoScramble(feats['expt']))
    
    pt, pp, pc, s = kfold(params, feats, b=b)

