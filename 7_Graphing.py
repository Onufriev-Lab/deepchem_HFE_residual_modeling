
from utils import *
from graphutils import *
import numpy as np
import pickle
import pandas as pd

x = list(np.arange(10))
y = list(np.abs(np.subtract(x, 5)))
y[3] += 0.99


pre_fill   = rgb2hex(hsv2rgb((0, 1, 1)))
pre_stroke = rgb2hex(hsv2rgb((0, 0.75, 0.6)))
post_fill   = rgb2hex(hsv2rgb((0.33, 1, 0.5)))
post_stroke = rgb2hex(hsv2rgb((0.33, 0.75, 0.3)))


w, h = 1500, 1500
dr = draw.Drawing(w, h)
dr.append(draw.Use(hist_svg(x, y, fill=pre_fill, stroke=pre_stroke), (w-1000)/2, (h-1000)/2))
dr.set_pixel_scale(2)
dr.save_svg("test.svg")


phy_models = ['tip3p', 'gbnsr6', 'igb5', 'asc', 'cha', 'null']
model_indx = [0, 0, 0, 0, 0, 0]

graphs = {}

for phy, indx in np.vstack((phy_models, model_indx)).T:
    graphs[phy] = {}

    run = pickle.load(open('tests/'+phy+'/model_'+str(indx)+'/results.pickle', 'rb'))
    runs = pickle.load(open('tests/'+phy+'/results.pickle', 'rb'))
    
    ###############################################   Scatter Plot   ###############################################
    w, h = 1300, 1300
    bw, bh = 1000, 1000
    sc_d = draw.Drawing(w, h)
    sc = draw.Group()
    sc_d.append(draw.Rectangle(-5, -5, w+10, h+10, fill=get_bgc()))
    o = [50, 30]
    
    #xr = [min(np.min(run['p_true'][0]['test']), np.min(run['p_true'][0]['test'])), max(np.max(run['p_true'][0]['test']), np.max(run['p_true'][0]['test']))]
    #yr = [min(np.min(run['p_phy'][0]['test']), np.min(run['p_corr'][0]['test'])), max(np.max(run['p_phy'][0]['test']), np.max(run['p_corr'][0]['test']))]
    #xr = list(np.add(xr, [-np.ptp(xr)*0.1, np.ptp(xr)*0.1]))
    #yr = list(np.add(yr, [-np.ptp(yr)*0.1, np.ptp(yr)*0.1]))
    xr = [-20.1, 4]
    yr = xr
    sc.append(draw.Line((w-bw)/2, (h-bh)/2+bh, (w-bw)/2+bw, (h-bh)/2, fill='none', opacity=0.5, stroke=fgc, stroke_width=16))

    sc.append(draw.Use(trim_svg(xr, yr, width=bw, height=bh, title=phy+" correction", xlabel=None, ylabel=None)
        , (w-bw)/2, (h-bh)/2))
    sc.append(draw.Use(scatter_svg(run['p_true'][0]['test'], run['p_phy'][0]['test'], xr=xr, yr=yr, width=bw, height=bh, r=15,
        fill=pre_fill, stroke=pre_stroke), (w-bw)/2, (h-bh)/2))
    sc.append(draw.Use(scatter_svg(run['p_true'][0]['test'], run['p_corr'][0]['test'], xr=xr, yr=yr, width=bw, height=bh, r=15,
        fill=post_fill, stroke=post_stroke), (w-bw)/2, (h-bh)/2))

    sc_d.append(draw.Use(sc, o[0], o[1]))
    sc_d.set_pixel_scale(2)
    sc_d.save_svg("graphs/scatter/"+phy+".svg")
    sc_d.save_png("graphs/scatter/"+phy+".png")
    graphs[phy]['scatter'] = sc

    
    ###############################################   Error Distribution   ###############################################
    w, h = 1300, 1300
    bw, bh = 1000, 1000
    ed_d = draw.Drawing(w, h)
    ed = draw.Group()
    ed_d.append(draw.Rectangle(-5, -5, w+10, h+10, fill=get_bgc()))
    o=[10, 30]

    xr=[-9, 5]
    yr=[0, 10]

    ed.append(draw.Use(trim_svg(xr, yr, width=bw, height=bh, title=phy+" error distribution", xlabel=None, ylabel=None, xticks=3, yticks=3, ygs=1)
        , (w-bw)/2, (h-bh)/2))

    x = run['p_phy'][0]['test']
    y = run['p_true'][0]['test']

    b = 30
    n, hx = np.histogram(np.subtract(x, y), bins=b, range=xr)
    n = n/len(x)*b
    ed.append(draw.Use(hist_svg((hx[:-1]+hx[1:])/2, n, width=bw, height=bh, xr=xr, yr=yr, fill=pre_fill, stroke=pre_stroke, alpha=0.75)
        , (w-bw)/2, (h-bh)/2))
    

    x = []
    y = []
    for i in range(len(runs['p_corr'])):
        x += runs['p_corr'][i]['test']
        y += runs['p_true'][i]['test']
    b = 100
    n, hx = np.histogram(np.subtract(x, y), bins=b, range=xr)
    n = n/len(x)*b
    ed.append(draw.Use(hist_svg((hx[:-1]+hx[1:])/2, n, width=bw, height=bh, xr=xr, yr=yr, fill=post_fill, stroke=post_stroke, alpha=0.5, s_scale=0.5
        ), (w-bw)/2, (h-bh)/2))
    
    ed_d.append(draw.Use(ed, o[0], o[1]))
    ed_d.set_pixel_scale(2)
    ed_d.save_svg("graphs/error/"+phy+"_error_hist.svg")
    ed_d.save_png("graphs/error/"+phy+"_error_hist.png")
    graphs[phy]['error'] = ed


    ###############################################   Correction Distribution   ###############################################
    w, h = 2300, 1300
    bw, bh = 2000, 1000
    cd_d = draw.Drawing(w, h)
    cd = draw.Group()
    cd_d.append(draw.Rectangle(-5, -5, w+10, h+10, fill=get_bgc()))
    o=[0, 30]

    xr=[-5, 10]
    yr=[0, 10]

    cd.append(draw.Use(trim_svg(xr, yr, width=bw, height=bh, title=phy+" correction distribution", xticks=10, yticks=2, xgs=0.5, ygs=1
        ), (w-bw)/2, (h-bh)/2))

    corr = []
    true = []
    phys = []
    for i in range(len(runs['p_corr'])):
        corr += runs['p_corr'][i]['test']
        true += runs['p_true'][i]['test']
        phys += runs['p_phy'][i]['test']
    
    b = 200
    impr = np.abs(np.subtract(phys, true))-np.abs(np.subtract(corr, true))
    n, hx = np.histogram(impr, bins=b, range=xr)
    n = n/len(true)*b

    cd.append(draw.Use(hist_svg((hx[:-1]+hx[1:])/2, n, width=bw, height=bh, xr=xr, yr=yr, fill=post_fill, stroke=post_stroke, alpha=0.5, s_scale=0.5),
        (w-bw)/2, (h-bh)/2))

    cd_d.append(draw.Use(cd, o[0], o[1]))
    cd_d.set_pixel_scale(2)
    cd_d.save_svg("graphs/correction/"+phy+"_correction_hist.svg")
    cd_d.save_png("graphs/correction/"+phy+"_correction_hist.png")
    graphs[phy]['corr'] = cd

    


###############################################   Consolidation   ###############################################

phy_models.pop(np.where(np.array(phy_models) == 'null')[0][0])
ws = [1300, 1300, 2600]
hs = list(np.ones(len(phy_models)))*1300
pad = 50

w, h = np.sum(np.add(ws, pad)), np.sum(np.add(hs, pad))
c = draw.Drawing(w, h)

yp = pad/2
for phy in graphs:
    xp = pad/2
    for i in range(len(graphs[phy].keys())):
        c.append(draw.Use(graphs[phy][list(graphs[phy].keys())[0]], xp, yp))
        xp += pad + ws[i]
    yp += pad + hs[0]

c.set_pixel_scale(1)
c.save_svg("graphs/consol.svg")
c.save_png("graphs/consol.png")



###############################################   Convergence Plot   ###############################################
w, h = 1500, 1500
bw, bh = 1000, 1000
con = draw.Drawing(w, h)
con.append(draw.Rectangle(-100, -100, w+200, h+200, fill=get_bgc()))

conv = pd.read_fwf("tests/convergence_log.txt", header=None).to_numpy()
xr = [-2, 502]
yr = [0, 2]
con.append(draw.Use(trim_svg(xr, yr, width=bw, height=bh, title='tip3p training convergence', xticks=3, yticks=3, xlabel='epochs', ylabel='loss'
), (w-bw)/2, (h-bh)/2))

for i in range(int(conv.shape[0]/500)):
    epoch = list(conv[500*(i  ):500*(i+1),0])
    loss  = list(np.clip(conv[500*(i  ):500*(i+1),1], yr[0], yr[1]))
    con.append(draw.Use(plot_svg(epoch, loss, xr=xr, yr=yr, width=bw, height=bh, s_scale=0.1, alpha=0.2, stroke=post_fill), (w-bw)/2, (h-bh)/2))

con.set_pixel_scale(2)
con.save_svg('graphs/other/convergence.svg')
con.save_png('graphs/other/convergence.png')
