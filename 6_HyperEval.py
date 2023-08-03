
from utils import *
import os
from shutil import rmtree

import_from('4_Kfold', ['pickle', 'psuedoScramble', 'partition', 'load_data', 'dc', 'runExp', 'Kfold', 'kfold'], __name__)


params = {'epochs' : 500, 'dropout' : 0.4, 'batch_normalize' : False, 'batch_size' : 100, 'feat' : 'tip3p', 'kfold' : 20, 'dense_layer_size' : 27, 'graph_conv_layers' : [53, 38], 'model_dir' : 'test.model', 'model_dir' : None}
phy_models = ['tip3p', 'gbnsr6', 'igb5', 'asc', 'null', 'zap9', 'cha']

def pick_average(models = phy_models):
    picks = {}
    phy_means = {}
    phy_stds = {}
    pick_stats = {}
    for phy in models:
        d = pickle.load(open('tests/'+phy+'/results.pickle', 'rb'))
        d['stats'][0].pop('params')
        means = dict([(stat, dict([(t, []) for t in d['stats'][0][stat].keys()])) for stat in d['stats'][0].keys()])
        stds = dict([(stat, dict([(t, 0) for t in d['stats'][0][stat].keys()])) for stat in d['stats'][0].keys()])
        for i in range(len(d['stats'])):
            for stat in means:
                for t in means[stat]:
                    means[stat][t] += [d['stats'][i][stat][t]]
        for stat in means:
            for t in means[stat]:
                stds[stat][t] = np.std(means[stat][t])
                means[stat][t] = np.mean(means[stat][t])
        phy_means[phy] = means
        phy_stds [phy] = stds
        picks[phy] = np.argmin(np.abs(np.subtract([d['stats'][i]['ml_rmsd']['train'] for i in range(len(d['stats']))], means['ml_rmsd']['train'])))
        pick_stats[phy] = d['stats'][picks[phy]].copy()
    #
    print("average stats")
    for t in means[list(means.keys())[0]].keys():
        print('#'*10, t, '#'*10)
        print(' '*17, end='')
        for phy in models:
            print((phy+' '*20)[:20], end='')
        print()
        for stat in d['stats'][0].keys():
            print((stat+' '*15)[:15], end='')
            for phy in models:
                print('& {u:.2f} $\\pm$ {s:.2f}        '.format(u=phy_means[phy][stat][t], s=phy_stds[phy][stat][t])[:20], end='')
            print('\n', end='')
        print('\n\n', end='')
    #
    print("pick stats")
    for t in means[list(means.keys())[0]].keys():
        print('#'*10, t, '#'*10)
        print(' '*17, end='')
        for phy in models:
            print((phy+' '*6)[:8], end='')
        print()
        for stat in d['stats'][0].keys():
            print((stat+' '*15)[:15], end='')
            for phy in models:
                print('& {:.2f}       '.format(pick_stats[phy][stat][t])[:8], end='')
            print('\n', end='')
        print('\n\n', end='')
    #
    return phy_means, picks

if __name__ == '__main__':
    feats = load_data()
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
            print(phy + " " + str(i))
            params['model_dir'] = 'model_' + str(i)
            if(os.path.isdir(params['model_dir'])):
                #print('skip')
                #continue
                rmtree(params['model_dir'])

            pt, pp, pc, s = kfold(params, feats, folds=folds, final=True)
            
            pickle.dump({'p_true' : pt, 'p_phy' : pp, 'p_corr' : pc, 'stats' : s, 'params' : params}, open(params['model_dir'] + '/results.pickle', 'wb'))
            p_true += pt
            p_phy += pp
            p_corr += pc
            stats += s

        pickle.dump({'p_true' : p_true, 'p_phy' : p_phy, 'p_corr' : p_corr, 'stats' : stats}, open('results.pickle', 'wb'))
        os.chdir(working_dir)

    os.chdir(os.path.dirname(os.getcwd()))


