
from utils import *
import os
from shutil import rmtree

import_from('1_Dictgen', ['pickle'], __name__)
import_from('4_Kfold', ['pickle', 'np', 'psuedoScramble', 'partition', 'load_data', 'dc', 'runExp', 'Kfold', 'kfold'], __name__)


params = {'epochs' : 500, 'dropout' : 0.4, 'batch_normalize' : False, 'batch_size' : 1000, 'feat' : 'tip3p', 'kfold' : 20, 'dense_layer_size' : 27, 'graph_conv_layers' : [32, 32], 'model_dir' : 'test.model', 'model_dir' : None}
feats = load_data()

params = {'epochs' : 500, 'dropout' : 0.4, 'batch_normalize' : False, 'batch_size' : 1000, 'feat' : 'tip3p', 'kfold' : -1, 'dense_layer_size' : 27, 'graph_conv_layers' : [32, 32], 'model_dir' : 'test.model'}
phy_models = ['tip3p', 'gbnsr6', 'igb5', 'asc', 'null', 'zap9', 'cha']

np.random.seed(10)
folds = Kfold(feats['expt'], params['kfold'])

soft_mkdir('tests')
os.chdir('tests')
working_dir = os.getcwd()

for phy in phy_models:
    soft_mkdir(phy)
    os.chdir(phy)
    params['feat'] = phy

    p_true, p_phy, p_corr, stats = [], [], [], []

    for i in range(100):
        params['model_dir'] = 'model_' + str(i)
        if(os.path.isfile(params['model_dir'])):
            rmtree(params['model_dir'])

        pt, pp, pc, s = kfold(params, feats, folds=folds, final=True)
        
        pickle.dump({'p_true' : pt, 'p_phy' : pp, 'p_corr' : pc, 'stats' : s}, open(params['model_dir'] + '/results.pickle', 'wb'))
        p_true += pt
        p_phy += pp
        p_corr += pc
        stats += s

    pickle.dump({'p_true' : p_true, 'p_phy' : p_phy, 'p_corr' : p_corr, 'stats' : stats}, open('results.pickle', 'wb'))
    os.chdir(working_dir)

os.chdir(os.path.dirname(os.getcwd()))


