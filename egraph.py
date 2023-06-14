
import pickle
import numpy as np
from utils import show
from matplotlib import pyplot as plt

params = [{'epochs' : 500, 'dropout' : 0.4, 'batch_normalize' : False, 'batch_size' : 100, 'feat' : 'null', 'kfold' : -1, 'dense_layer_size' : 27, 'graph_conv_layers' : [32, 32]},
          {'epochs' : 500, 'dropout' : 0.4, 'batch_normalize' : False, 'batch_size' : 100, 'feat' : 'tip3p', 'kfold' : -1, 'dense_layer_size' : 27, 'graph_conv_layers' : [32, 32]},
          {'epochs' : 500, 'dropout' : 0.4, 'batch_normalize' : False, 'batch_size' : 100, 'feat' : 'gbnsr6', 'kfold' : -1, 'dense_layer_size' : 27, 'graph_conv_layers' : [32, 32]},
          {'epochs' : 500, 'dropout' : 0.4, 'batch_normalize' : False, 'batch_size' : 100, 'feat' : 'igb5', 'kfold' : -1, 'dense_layer_size' : 27, 'graph_conv_layers' : [32, 32]},
          {'epochs' : 500, 'dropout' : 0.4, 'batch_normalize' : False, 'batch_size' : 100, 'feat' : 'asc', 'kfold' : -1, 'dense_layer_size' : 27, 'graph_conv_layers' : [32, 32]},
          {'epochs' : 500, 'dropout' : 0.4, 'batch_normalize' : False, 'batch_size' : 100, 'feat' : 'zap9', 'kfold' : -1, 'dense_layer_size' : 27, 'graph_conv_layers' : [32, 32]},
          {'epochs' : 500, 'dropout' : 0.4, 'batch_normalize' : False, 'batch_size' : 100, 'feat' : 'cha', 'kfold' : -1, 'dense_layer_size' : 27, 'graph_conv_layers' : [32, 32]}]

stats, points = [], []
for p in params:
    stats.append(pickle.load(open('runs/final_' + p['feat'] + '_dropout_' + str(p['dropout']) + '_dense_' + str(p['dense_layer_size']), 'rb')))
    points.append(pickle.load(open('runs/final_' + p['feat'] + '_dropout_' + str(p['dropout']) + '_dense_' + str(p['dense_layer_size']) + '_points', 'rb')))
    
    stats[-1]['null->phy'] = {'train':          1 - np.divide(np.power(stats[-1]['phy_rmsd']['train'], 2), np.power(stats[ 0]['phy_rmsd']['train'], 2))
                            , 'test' :          1 - np.divide(np.power(stats[-1]['phy_rmsd']['test' ], 2), np.power(stats[ 0]['phy_rmsd']['test' ], 2))}
    stats[-1]['phy->ml']   = {'train':          1 - np.divide(np.power(stats[-1][ 'ml_rmsd']['train'], 2), np.power(stats[-1]['phy_rmsd']['train'], 2))
                            , 'test' :          1 - np.divide(np.power(stats[-1][ 'ml_rmsd']['test' ], 2), np.power(stats[-1]['phy_rmsd']['test' ], 2))}
    stats[-1]['null->ml']  = {'train':          1 - np.divide(np.power(stats[-1][ 'ml_rmsd']['train'], 2), np.power(stats[ 0]['phy_rmsd']['train'], 2))
                            , 'test' :          1 - np.divide(np.power(stats[-1][ 'ml_rmsd']['test' ], 2), np.power(stats[ 0]['phy_rmsd']['test' ], 2))}
scale=1

phyr_pts = [] # final physics rmsd
mlr_pts = [] # final ml rmsd

for i in range(len(params)):
    print(params[i]['feat'])
    show(stats[i], params[i]['feat'])
    phyr_pts.append(np.mean(stats[i]['phy_rmsd']['test']))
    mlr_pts.append(np.mean(stats[i]['ml_rmsd']['test']))
    ###############################################   Scatter Plot   ###############################################
    plt.gcf().set_size_inches(5*scale, 5*scale)
    plt.title(params[i]['feat'] + " correction")
    plt.xlabel('experimental solvation (kcal/mol)')
    plt.ylabel('model solvation (kcal/mol)')
    
    if(params[i]['feat'] != 'null'):
        plt.scatter(points[i]['p_true']['test'][:80], points[i]['p_phy' ]['test'][:80], s=50, c = 'red'  , edgecolor='red'   , label=params[i]['feat'], marker='.')
    plt.scatter(points[i]['p_true']['test'][:80], points[i]['p_corr']['test'][:80], s=50, c = 'green', edgecolor= 'green', label=params[i]['feat'] + " + ML" , marker='.')
    
    bfit = np.linalg.lstsq(
        np.vstack((points[i]['p_true']['train'], np.ones(len(points[i]['p_true']['train'])))).T,
        np.matrix(points[i]['p_phy' ]['train']).T,
        rcond=None)
    afit = np.linalg.lstsq(
        np.vstack((points[i]['p_true']['train'], np.ones(len(points[i]['p_true']['train'])))).T,
        np.matrix(points[i]['p_corr' ]['train']).T,
        rcond=None)
    r = [-20, 5]
    if(params[i]['feat'] != 'null'):
        plt.legend()
        plt.plot(r, np.asarray(np.matmul(np.vstack((r, np.ones(len(r)))).T, bfit[0]).T).flatten(), c='red', label=params[i]['feat'])
    else:
        plt.title('pure ML model')
    plt.plot(r, np.asarray(np.matmul(np.vstack((r, np.ones(len(r)))).T, afit[0]).T).flatten(), c='green', label=params[i]['feat'] + " + ML")
    #print(type(bfit[0][0, 0]))
    if(params[i]['feat'] != 'null'):
        plt.text(-10, -17, 
            'y = ' + "{: .2f}".format(bfit[0][0, 0]) + " x + " + "{: .2f}".format(bfit[0][1, 0]) #+ 
            #'  r = ' + "{: .3f}".format(np.sqrt(bfit[1][0, 0]/len(points[i]['p_true']['train'])))
            , c='red')
    plt.text(-10, -18, 
        'y = ' + "{: .2f}".format(afit[0][0, 0]) + " x + " + "{: .2f}".format(afit[0][1, 0]) #+ 
        #'  r = ' + "{: .3f}".format(np.sqrt(afit[1][0, 0]/len(points[i]['p_true']['train'])))
        , c='green')


    plt.plot(r, r, linestyle='dotted', linewidth=1, c='black')
    plt.xlim(r)
    plt.ylim(r)
    plt.savefig('runs/graphs/scatter/' + params[i]['feat'] + '.svg', dpi = 300)

    ###############################################   Error Distribution   ###############################################
    plt.clf()
    plt.title(params[i]['feat'] + ' error distribution')
    plt.xlabel('model error (kcal/mol)')
    #plt.hist(np.array(points[i]['p_phy']['test']) - np.array(points[i]['p_true']['test']), label=params[i]['feat'], bins = 25)
    #plt.hist(np.array(points[i]['p_corr']['test']) - np.array(points[i]['p_true']['test']), label=params[i]['feat'] + '+ML', bins = 25, alpha=0.5)
    r = [-9, 5]
    plt.xlim(r)
    plt.ylim([0, 10])
    b = 30
    n, x = np.histogram(np.array(points[i]['p_phy']['test']) - np.array(points[i]['p_true']['test']), bins=b, range=r)
    n = n/len(points[i]['p_true']['test'])*b
    if(params[i]['feat'] != 'null'):
        plt.bar((x[:-1]+x[1:])/2, n, edgecolor='none', width=(r[1]-r[0])/(b-1), label=params[i]['feat'], color='red')
    else:
        plt.title('pure ML model' + ' error distribution')
        
    b = 100
    n, x = np.histogram(np.array(points[i]['p_corr']['test']) - np.array(points[i]['p_true']['test']), bins = b, range=r)
    n = n/len(points[i]['p_true']['test'])*b
    plt.bar((x[:-1]+x[1:])/2, n, edgecolor='none', width=(r[1]-r[0])/(b-0), label=params[i]['feat']+'+ML', alpha=0.85, color='green', rasterized=True)
    
    if(params[i]['feat'] != 'null'):
        plt.legend()
    
    plt.grid(axis='x', alpha = 0.2, color=(0, 0, 0), linewidth=0.5)
    plt.savefig('runs/graphs/error/'+ params[i]['feat'] + '_error_hist.svg', dpi = 1000)
    
    ###############################################   Correction Distribution   ###############################################
    plt.clf()
    plt.gcf().set_size_inches(7*scale, 2*scale)
    plt.title(params[i]['feat'] + ' correction distribution')
    r = [-5, 10]
    plt.xlim(r)
    plt.ylim([-0.2, 10])
    b = 200
    corr = np.abs(np.array(points[i]['p_phy']['test']) - np.array(points[i]['p_true']['test']))-np.abs(np.array(points[i]['p_corr']['test']) - np.array(points[i]['p_true']['test']))
    n, x = np.histogram(corr, bins = b, range=r)
    
    n = n/len(points[i]['p_true']['test'])*b
    #n = n+(n>0)*0.05
    plt.bar((x[:-1]+x[1:])/2, n, edgecolor='none', width=(r[1]-r[0])/(b-50), color = 'green')
    plt.bar((x[:-1]+x[1:])/2, -1*(n>0), edgecolor='none', width=(r[1]-r[0])/(b-50), color = 'green')

    plt.xlabel('ML improvement over ' + params[i]['feat'] + ' (kcal/mol)')
    plt.grid(axis='x', alpha=0.2, color=(0, 0, 0), linewidth=0.5)

    plt.plot([0, 0], [-10, 10], color=(0, 0, 0), linewidth=1.5, alpha=0.75)
    plt.plot([max(corr), max(corr)], [-10, 10], color=(1, 0, 0), linewidth=1.5, alpha=0.5)
    plt.plot([min(corr), min(corr)], [-10, 10], color=(1, 0, 0), linewidth=1.5, alpha=0.5)

    plt.text(6, 5, r'$\mu$'+" = "+"{: .3f}".format(np.mean(corr)))
    plt.text(6, 4, r'$\sigma$'+" = "+"{: .3f}".format(np.std(corr)))

    plt.savefig('runs/graphs/correction/'+ params[i]['feat'] + '_correction_hist.svg', dpi = 300)
    plt.clf()

r = [0, 4]
fit = np.linalg.lstsq(
        np.vstack((phyr_pts, np.ones(len(phyr_pts)))).T,
        np.matrix(mlr_pts).T,
        rcond=None)
#plt.plot(r, np.asarray(np.matmul(np.vstack((r, np.ones(len(r)))).T, fit[0]).T).flatten(), c='green', linewidth=0.7)

plt.gcf().set_size_inches(4.5*scale, 4.5*scale)
plt.xlim(r)
plt.ylim(r)
plt.xlabel("physics RMSD")
plt.ylabel("ML + physics RMSD")
plt.title("Physics + ML improvement")

plt.scatter(phyr_pts, mlr_pts, color='green', s=32)
plt.savefig('runs/graphs/other/improvement1.svg', dpi=300)
plt.clf()

mlr_pts = np.divide((np.array(phyr_pts)-np.array(mlr_pts)), phyr_pts)*100
fit = np.linalg.lstsq(
        np.vstack((phyr_pts, np.ones(len(phyr_pts)))).T,
        np.matrix(mlr_pts).T,
        rcond=None)
#plt.plot(r, np.asarray(np.matmul(np.vstack((r, np.ones(len(r)))).T, fit[0]).T).flatten(), c='green', linewidth=0.7)

plt.gcf().set_size_inches(4.5*scale, 4.5*scale)
plt.xlim(r)
plt.ylim([0, 100])
plt.xlabel("physics RMSD")
plt.ylabel("ML -> physics % improvement")
plt.title(r"Physics + ML $\Delta$ improvement")

plt.scatter(phyr_pts, mlr_pts, color='green', s=32)
plt.savefig('runs/graphs/other/improvement2.svg', dpi=300)
