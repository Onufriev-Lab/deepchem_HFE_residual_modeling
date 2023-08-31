
from utils import *
import os
from shutil import rmtree

import_from('4_Kfold', ['pickle', 'psuedoScramble', 'partition', 'load_data', 'dc', 'runExp', 'Kfold', 'kfold'], __name__)


default_params = {'epochs' : 500, 'dropout' : 0.4, 'batch_normalize' : False, 'batch_size' : 100, 'feat' : 'tip3p', 'kfold' : 20, 'dense_layer_size' : 27, 'graph_conv_layers' : [32, 32], 'model_dir' : 'test.model', 'model_dir' : None}

def get_t1_params():
    params = default_params.copy()
    t1_params = []
    dl_arr = np.array(10*np.power(1.65, np.arange(0, 10, 1)), int)
    drop_arr = np.arange(0, 1, 0.1)
    for dl in dl_arr:
        for drop in drop_arr:
            params['dropout'] = drop
            params['dense_layer_size'] = dl
            yield params

def load_t1():
    data = []
    for i in range(10):
        data.append([None]*10)
    dl_arr = np.array(10*np.power(1.65, np.arange(0, 10, 1)), int)
    drop_arr = np.arange(0, 1, 0.1)

    for p in get_t1_params():
        d = pickle.load(open('hypertests/dense_dropout/'+gen_dir_name(p)+'/results.pickle', 'rb'))
        data[np.where(dl_arr == d['stats'][0]['params']['dense_layer_size'])[0][0]][np.where(drop_arr == d['stats'][0]['params']['dropout'])[0][0]] = d

    return data

def t1_results():
    data = load_t1()
    train = np.zeros((len(data), len(data[0])))
    valid = np.zeros((len(data), len(data[0])))
    k = []
    for i in range(len(data)):
        k.append([None]*len(data[0]))
    
    for y in range(len(data)):
        for x in range(len(data[y])):
            #if(len(data[y][x]['stats']) != 100):
            #    print("missing ", x, y)
            for n in range(len(data[y][x]['stats'])):
                train[y, x] += data[y][x]['stats'][n]['ml_rmsd']['train']
                valid[y, x] += data[y][x]['stats'][n]['ml_rmsd']['valid']
            train[y, x] *= 1.0/len(data[y][x]['stats'])
            valid[y, x] *= 1.0/len(data[y][x]['stats'])
            k[y][x] =  {'dl' : data[y][x]['stats'][0]['params']['dense_layer_size'], 'drop' : data[y][x]['stats'][0]['params']['dropout']}

    return train, valid, valid-train, k

def get_t2_params():
    params = default_params.copy()
    t2_params = []
    cl_arr = np.floor(np.power(np.arange(0, 1, 0.1), 1.8)*128)+2
    for c1 in cl_arr:
        for c2 in cl_arr:
            params['graph_conv_layers'] = [int(c1), int(c2)]
            yield params

def load_t2():
    data = []
    for i in range(10):
        data.append([None]*10)
    cl_arr = np.floor(np.power(np.arange(0, 1, 0.1), 1.8)*128)+2
    #
    for p in get_t2_params():
        d = pickle.load(open('hypertests/cl_0.3/'+gen_dir_name(p)+'/results.pickle', 'rb'))
        data[np.where(cl_arr == d['stats'][0]['params']['graph_conv_layers'][0])[0][0]][np.where(cl_arr == d['stats'][0]['params']['graph_conv_layers'][1])[0][0]] = d
    #
    return data

def t2_results():
    data = load_t2()
    train = np.zeros((len(data), len(data[0])))
    valid = np.zeros((len(data), len(data[0])))
    k = []
    for i in range(len(data)):
        k.append([None]*len(data[0]))
    #
    for y in range(len(data)):
        for x in range(len(data[y])):
            #if(len(data[y][x]['stats']) != 40):
            #    print("missing ", x, y, len(data[y][x]['stats']))
            for n in range(len(data[y][x]['stats'])):
                train[y, x] += data[y][x]['stats'][n]['ml_rmsd']['train']
                valid[y, x] += data[y][x]['stats'][n]['ml_rmsd']['valid']
            train[y, x] *= 1.0/len(data[y][x]['stats'])
            valid[y, x] *= 1.0/len(data[y][x]['stats'])
            k[y][x] =  {'c1' : data[y][x]['stats'][0]['params']['graph_conv_layers'][0], 'c2' : data[y][x]['stats'][0]['params']['graph_conv_layers'][1]}
    #
    return train, valid, valid-train, k

def gen_dir_name(p):
    return "drop_{dropout:.1f}_dense_{dense_layer_size:.0f}_c1_{graph_conv_layers[0]:.0f}_c2_{graph_conv_layers[1]:.0f}".format(**p)

def lr():
    ps = os.listdir('hypertests/cl_0.3/')
    place=[len(ps)]
    for p in get_t2_params():
        if(gen_dir_name(p) in ps):
            pr = p.copy()
    ms = np.sort(os.listdir('hypertests/cl_0.3/'+gen_dir_name(pr)))
    place.append(len(ms))
    ks = np.sort(os.listdir('hypertests/cl_0.3/'+gen_dir_name(pr)+'/'+ms[-1]))
    place.append(len(ks))
    return place

def cut(k, cutoff=0.3):
    train, valid, diff, k = k
    m = np.argmin(valid+(diff>=cutoff)*100)
    print('valid : {valid:.2f}\ndiff : {diff:.2f}\n'.format(valid=np.array(valid).flatten()[m], diff=np.array(diff).flatten()[m]), np.array(k).flatten()[m])



if __name__ == '__main__':
    feats = load_data()

    np.random.seed(10)
    folds = Kfold(feats['expt'], default_params['kfold'])

    parent_dir = os.getcwd()
    soft_mkdir('hypertests')
    os.chdir('hypertests')
    soft_mkdir('cl_0.3')
    os.chdir('cl_0.3')
    working_dir = os.getcwd()

    for p in get_t2_params():
        os.chdir(working_dir)
        trial_dir = gen_dir_name(p)
        #if (os.path.isdir(trial_dir)):
        #    print("skipped: " + trial_dir)
        #    continue
        soft_mkdir(trial_dir)
        os.chdir(trial_dir)

        p_true, p_phy, p_corr, stats = [], [], [], []
        if(os.path.isfile('results.pickle')):
            print("appended " + trial_dir)
            prev = pickle.load(open('results.pickle', 'rb'))
            p_true, p_phy, p_corr, stats = prev['p_true'], prev['p_phy'], prev['p_corr'], prev['stats']


        for i in range(5):
            print(gen_dir_name(p) + " " + str(i))
            p['model_dir'] = 'model_' + str(i)
            if(os.path.isdir(p['model_dir'])):
                print('skip '+ p['model_dir'])
                continue
                #rmtree(p['model_dir'])

            pt, pp, pc, s = kfold(p, feats, folds=folds, final=False)
            
            pickle.dump({'p_true' : pt, 'p_phy' : pp, 'p_corr' : pc, 'stats' : s, 'params' : p}, open('model_' + str(i) + '/results.pickle', 'wb'))
            p_true += pt
            p_phy += pp
            p_corr += pc
            stats += s

        pickle.dump({'p_true' : p_true, 'p_phy' : p_phy, 'p_corr' : p_corr, 'stats' : stats, 'params' : p}, open('results.pickle', 'wb'))
        

    os.chdir(parent_dir)


