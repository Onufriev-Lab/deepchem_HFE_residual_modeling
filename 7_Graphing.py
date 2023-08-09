
from utils import *
from graphutils import *
import numpy as np
import pickle
import pandas as pd

import_from('5_HyperExp', ['t1_results', 't2_results'], __name__)
import_from('6_HyperEval', ['pick_average'], __name__)

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


#phy_models = ['tip3p', 'gbnsr6', 'asc', 'igb5', 'cha', 'null']
phy_models = ['tip3p', 'asc', 'cha', 'gbnsr6', 'igb5', 'null']

disp_names = {'tip3p' : 'TIP3P', 'gbnsr6' : 'GBNSR6', 'asc' : 'AASC', 'igb5' : 'IGB5', 'cha' : 'CHA-GB', 'null' : 'null'}
_, model_indx = pick_average(phy_models)

graphs = {}

for phy in phy_models:
    graphs[phy] = {}
    indx = int(model_indx[phy])
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

    sc.append(draw.Use(trim_svg(xr, yr, width=bw, height=bh, title=disp_names[phy]+" correction", xlabel=None, ylabel=None)
            , (w-bw)/2, (h-bh)/2))
    
    if(phy != 'null'):
        sc.append(draw.Use(scatter_svg(run['p_true'][0]['test'], run['p_phy'][0]['test'], xr=xr, yr=yr, width=bw, height=bh, r=15,
            fill=pre_fill, stroke=pre_stroke), (w-bw)/2, (h-bh)/2))
    sc.append(draw.Use(scatter_svg(run['p_true'][0]['test'], run['p_corr'][0]['test'], xr=xr, yr=yr, width=bw, height=bh, r=15,
        fill=post_fill, stroke=post_stroke), (w-bw)/2, (h-bh)/2))

    xrd =  np.arange(xr[0], xr[1], (xr[1]-xr[0])/50)
    bfit = np.linalg.lstsq(
        np.vstack((run['p_true'][0]['test'], np.ones(len(run['p_true'][0]['test'])))).T,
        np.matrix(run['p_phy' ][0]['test']).T,
        rcond=None)
    bfit = list(np.asarray(np.matmul(np.vstack((xrd, np.ones(len(xrd)))).T, bfit[0]).T).flatten())

    for i in range(0, len(xrd), 2):
        if(bfit[i] > yr[0] and bfit[i] < yr[1] and bfit[i+1] > yr[0] and bfit[i+1] < yr[1]):
            sc.append(draw.Line(mapRange(xr[0], xr[1], 0, bw, xrd[i  ])+(w-bw)/2, mapRange(yr[0], yr[1], bh, 0, bfit[i  ])+(h-bh)/2,
                                mapRange(xr[0], xr[1], 0, bw, xrd[i+1])+(w-bw)/2, mapRange(yr[0], yr[1], bh, 0, bfit[i+1])+(h-bh)/2,
                                stroke=pre_stroke, fill='none', stroke_width=8, stroke_opacity=0.75))

    afit = np.linalg.lstsq(
        np.vstack((run['p_true'][0]['test'], np.ones(len(run['p_true'][0]['test'])))).T,
        np.matrix(run['p_corr' ][0]['test']).T,
        rcond=None)
    afit = list(np.asarray(np.matmul(np.vstack((xrd, np.ones(len(xrd)))).T, afit[0]).T).flatten())


    for i in range(0, len(xrd), 2):
        if(afit[i] > yr[0] and afit[i] < yr[1] and afit[i+1] > yr[0] and afit[i+1] < yr[1]):
            sc.append(draw.Line(mapRange(xr[0], xr[1], 0, bw, xrd[i  ])+(w-bw)/2, mapRange(yr[0], yr[1], bh, 0, afit[i  ])+(h-bh)/2,
                                mapRange(xr[0], xr[1], 0, bw, xrd[i+1])+(w-bw)/2, mapRange(yr[0], yr[1], bh, 0, afit[i+1])+(h-bh)/2,
                                stroke=post_stroke, fill='none', stroke_width=8, stroke_opacity=0.75))

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

    ed.append(draw.Use(trim_svg(xr, yr, width=bw, height=bh, title=disp_names[phy]+" error distribution", xlabel=None, ylabel=None, xticks=3, yticks=3, xgs=1, ygs=1)
        , (w-bw)/2, (h-bh)/2))

    x = run['p_phy'][0]['test']
    y = run['p_true'][0]['test']

    b = 30
    n, hx = np.histogram(np.subtract(x, y), bins=b, range=xr)
    n = n/len(x)*b

    if(phy != 'null'):
        ed.append(draw.Use(hist_svg((hx[:-1]+hx[1:])/2, n, width=bw, height=bh, xr=xr, yr=yr, fill=pre_fill, stroke=pre_stroke, alpha=0.75)
            , (w-bw)/2, (h-bh)/2))
    

    x = []
    y = []
    #for i in range(len(runs['p_corr'])):
    #    x += runs['p_corr'][i]['test']
    #    y += runs['p_true'][i]['test']
    x += run['p_corr'][0]['test']
    y += run['p_true'][0]['test']

    b = 30#100
    n, hx = np.histogram(np.subtract(x, y), bins=b, range=xr)
    n = n/len(x)*b
    ed.append(draw.Use(hist_svg((hx[:-1]+hx[1:])/2, n, width=bw, height=bh, xr=xr, yr=yr, fill=post_fill, stroke=post_stroke, alpha=0.75, s_scale=0.5
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
    yr=[0, 15]

    cd.append(draw.Use(trim_svg(xr, yr, width=bw, height=bh, title=disp_names[phy]+" improvement distribution", xticks=10, yticks=2, xgs=0.5, ygs=1
        ), (w-bw)/2, (h-bh)/2))

    corr = []
    true = []
    phys = []
    #for i in range(len(runs['p_corr'])):
    #    corr += runs['p_corr'][i]['test']
    #    true += runs['p_true'][i]['test']
    #    phys += runs['p_phy'][i]['test']
    corr += run['p_corr'][0]['test']
    true += run['p_true'][0]['test']
    phys += run['p_phy'][0]['test']
    
    b = 30#200
    impr = np.abs(np.subtract(phys, true))-np.abs(np.subtract(corr, true))
    n, hx = np.histogram(impr, bins=b, range=xr)
    n = n/len(true)*b

    cd.append(draw.Use(hist_svg((hx[:-1]+hx[1:])/2, n, width=bw, height=bh, xr=xr, yr=yr, fill=post_fill, stroke=post_stroke, alpha=0.85, s_scale=0.75),
        (w-bw)/2, (h-bh)/2))

    cd.append(draw.Line(mapRange(xr[0], xr[1], 0, bw, 0)+(w-bw)/2, (h-bh)/2, mapRange(xr[0], xr[1], 0, bw, 0)+(w-bw)/2, bh+(h-bh)/2, stroke_width=20, stroke_opacity=0.5, stroke=get_fgc()))
    
    arrow = draw.Marker(-0.002, -0.51, 0.9, 0.5, scale=4, orient='auto')
    arrow.append(draw.Lines(-0.1, 0.5, -0.1, -0.5, 0.9, 0, fill=pre_stroke, close=True, opacity=0.5))

    #impr_min = np.min(impr)
    #cd.append(draw.Line(mapRange(xr[0], xr[1], 0, bw, impr_min)+(w-bw)/2, bh*0.75+(h-bh)/2, mapRange(xr[0], xr[1], 0, bw, impr_min)+(w-bw)/2, bh+(h-bh)/2-70, stroke_width=20, stroke_opacity=0.5, stroke=pre_stroke, marker_end=arrow))
    #cd.append(draw.Text("max-", 100, mapRange(xr[0], xr[1], 0, bw, impr_min)+(w-bw)/2, bh*0.75+(h-bh)/2-60, fill=pre_stroke, opacity=0.75, text_anchor='middle', font_family='Arial'))
    #impr_max = np.max(impr)
    #cd.append(draw.Line(mapRange(xr[0], xr[1], 0, bw, impr_max)+(w-bw)/2, bh*0.75+(h-bh)/2, mapRange(xr[0], xr[1], 0, bw, impr_max)+(w-bw)/2, bh+(h-bh)/2-70, stroke_width=20, stroke_opacity=0.5, stroke=pre_stroke, marker_end=arrow))
    #cd.append(draw.Text("max+", 100, mapRange(xr[0], xr[1], 0, bw, impr_max)+(w-bw)/2, bh*0.75+(h-bh)/2-60, fill=pre_stroke, opacity=0.75, text_anchor='middle', font_family='Arial'))
    
    cd.append(draw.Text('mean'+" = "+'{:.2f}'.format(np.mean(impr)), 100, bw*0.75+(w-bw)/2, bh*0.25+(h-bh)/2-40, fill=post_stroke, opacity=0.75, text_anchor='end', font_family='Arial'))
    cd.append(draw.Text('std'+" = "+'{:.2f}'.format(np.std(impr)), 100, bw*0.75+(w-bw)/2, bh*0.25+(h-bh)/2+60, fill=post_stroke, opacity=0.75, text_anchor='end', font_family='Arial'))


    cd_d.append(draw.Use(cd, o[0], o[1]))
    cd_d.set_pixel_scale(2)
    cd_d.save_svg("graphs/correction/"+phy+"_correction_hist.svg")
    cd_d.save_png("graphs/correction/"+phy+"_correction_hist.png")
    graphs[phy]['corr'] = cd

    


###############################################   Consolidation   ###############################################

phy_models.pop(-1)
ws = [1500, 1300, 2300]
hs = list(np.ones(len(phy_models))*1300)
pad = 50

w, h = np.sum(np.add(ws, pad)), np.sum(np.add(hs, pad))
c_d = draw.Drawing(w+200, h+200)
c = draw.Group()
c_d.append(draw.Rectangle(-10, -10, c_d.width+20, c_d.height+20, fill=get_bgc()))

yp = pad/2
for phy in phy_models:
    xp = pad/2+20
    for i in range(len(graphs[phy].keys())):
        #c.append(draw.Rectangle(xp, yp, ws[i], hs[0], fill=rgb2hex(hsv2rgb((0.66, 0.02, 0.95)))))
        c.append(draw.Use(graphs[phy][list(graphs[phy].keys())[i]], xp, yp))
        xp += pad + ws[i]
    yp += pad + hs[0]

c_d.append(draw.Text('experiment (kcal/mol)', 125, 300+pad/2+1300/2, yp+50, text_anchor='middle', fill=get_fgc(), opacity=0.75, font_family='Arial'))
c_d.append(draw.Text('model error (kcal/mol)', 125, 300+pad/2+1500+pad+1300/2+50, yp+50, text_anchor='middle', fill=get_fgc(), opacity=0.75, font_family='Arial'))
c_d.append(draw.Text('ML improvement (kcal/mol)', 125, 300+pad/2+1500+pad+1300+pad+2300/2, yp+50, text_anchor='middle', fill=get_fgc(), opacity=0.75, font_family='Arial'))

txt = draw.Text('model (kcal/mol)', 125, -yp+650, 200, transform='rotate(-90)', text_anchor='middle', fill=get_fgc(), opacity=0.75, font_family='Arial')
txtg = draw.Group()
txtg.append(txt)
c_d.append(txtg)

txt = draw.Text('probability density', 125, -yp+650, 200+pad+1500+60, transform='rotate(-90)', text_anchor='middle', fill=get_fgc(), opacity=0.75, font_family='Arial')
txtg = draw.Group()
txtg.append(txt)
c_d.append(txtg)

txt = draw.Text(' = before ML correction', 125, -yp+650+1200, 200, transform='rotate(-90)', text_anchor='start', fill=get_fgc(), opacity=0.75, font_family='Arial')
txtg = draw.Group()
txtg.append(txt)
c_d.append(txtg)

txt = draw.Text(' = after ML correction', 125, -yp+650+1200+1800, 200, transform='rotate(-90)', text_anchor='start', fill=get_fgc(), opacity=0.75, font_family='Arial')
txtg = draw.Group()
txtg.append(txt)
c_d.append(txtg)

c_d.append(draw.Rectangle(120, 90+yp-650-1200     , 80, 120, fill=pre_fill , stroke=pre_stroke , stroke_width=15, opacity=0.75, stroke_opacity=0.75))
c_d.append(draw.Rectangle(120, 90+yp-650-1200-1800, 80, 120, fill=post_fill, stroke=post_stroke, stroke_width=15, opacity=0.75, stroke_opacity=0.75))

c_d.append(draw.Use(c, 300, 0))
c_d.set_pixel_scale(1)
c_d.save_svg("graphs/consol.svg")
c_d.save_png("graphs/consol.png")



###############################################   Convergence Plot   ###############################################
w, h = 1500, 1500
bw, bh = 1000, 1000
con = draw.Drawing(w, h)
con.append(draw.Rectangle(-100, -100, w+200, h+200, fill=get_bgc()))

conv = pd.read_fwf("tests/convergence_log.txt", header=None).to_numpy()
xr = [-2, 502]
yr = [0, 2]
con.append(draw.Use(trim_svg(xr, yr, width=bw, height=bh, title=disp_names['tip3p'] + ' training convergence', xticks=3, yticks=3, xlabel='epochs', ylabel='loss'
), (w-bw)/2, (h-bh)/2))

for i in range(int(conv.shape[0]/500)):
    epoch = list(conv[500*(i  ):500*(i+1),0])
    loss  = list(np.clip(conv[500*(i  ):500*(i+1),1], yr[0], yr[1]))
    con.append(draw.Use(plot_svg(epoch, loss, xr=xr, yr=yr, width=bw, height=bh, s_scale=0.1, alpha=0.2, stroke=post_fill), (w-bw)/2, (h-bh)/2))

con.set_pixel_scale(2)
con.save_svg('graphs/other/convergence.svg')
con.save_png('graphs/other/convergence.png')


###############################################   Null Plot   ###############################################

sc = graphs['null']['scatter']
ed = graphs['null']['error']

w, h = 1500, 1500

sc_d = draw.Drawing(w, h)
ed_d = draw.Drawing(w, h)
sc_d.append(draw.Rectangle(-10, -10, sc_d.width+20, sc_d.height+20, fill=get_bgc()))
ed_d.append(draw.Rectangle(-10, -10, ed_d.width+20, ed_d.height+20, fill=get_bgc()))

sc.append(draw.Text('experiment (kcal/mol)', 110, 1300/2, 1300+50, text_anchor='middle', font_family='Arial', fill=get_fgc(), opacity=0.75))
ed.append(draw.Text('model error (kcal/mol)' , 110, 1300/2, 1300+50, text_anchor='middle', font_family='Arial', fill=get_fgc(), opacity=0.75))

txt = draw.Text('model (kcal/mol)', 125, -1300/2, -50, transform='rotate(-90)', text_anchor='middle', fill=get_fgc(), opacity=0.75, font_family='Arial')
txtg = draw.Group()
txtg.append(txt)
sc.append(txtg)

txt = draw.Text('probability density', 125, -1300/2, -50, transform='rotate(-90)', text_anchor='middle', fill=get_fgc(), opacity=0.75, font_family='Arial')
txtg = draw.Group()
txtg.append(txt)
ed.append(txtg)

sc_d.append(draw.Use(sc, 200, 100))
ed_d.append(draw.Use(ed, 200, 100))
sc_d.set_pixel_scale(2)
ed_d.set_pixel_scale(2)
sc_d.save_svg('graphs/scatter/null.svg')
ed_d.save_svg('graphs/error/null_error_hist.svg')
sc_d.save_png('graphs/scatter/null.png')
ed_d.save_png('graphs/error/null_error_hist.png')

###############################################   improvement Plot   ###############################################

x, y, xp, yp = [], [], [], []
#phy_models.append('null')
for phy in phy_models:
    runs = pickle.load(open('tests/'+phy+'/results.pickle', 'rb'))
    m_phy_test, m_phy_train = [], []
    m_ml_test,  m_ml_train  = [], []
    for i in runs['stats']:
        m_phy_test.append(i['phy_rmsd']['test'])
        m_ml_test.append(i['ml_rmsd']['test'])
        m_phy_train.append(i['phy_rmsd']['train'])
        m_ml_train.append(i['ml_rmsd']['train'])
    x .append(m_phy_test[int(model_indx[phy])])#np.mean(m_phy_test))
    y .append(m_ml_test[int(model_indx[phy])])#np.mean(m_ml_test))
    xp.append(m_phy_train[int(model_indx[phy])])#np.mean(m_phy_train))
    yp.append(m_ml_train[int(model_indx[phy])])#np.mean())



bw, bh = 1000, 1000
w, h = 1500, 1500
vs_d = draw.Drawing(w, h)
de_d = draw.Drawing(w, h)
vs_d.append(draw.Rectangle(-10, -10, vs_d.width+20, vs_d.height+20, fill=get_bgc()))
de_d.append(draw.Rectangle(-10, -10, de_d.width+20, de_d.height+20, fill=get_bgc()))

vs = draw.Group()
de = draw.Group()

xr = [0, 4]
yr = [0, 4]

vs.append(draw.Use(trim_svg(xr, yr, width=bw, height=bh, title='absolute improvement', xlabel='physics model RMSE (kcal/mol)', ylabel='physics + ML RMSE (kcal/mol)')
    , (w-bw)/2, (h-bh)/2))


xrd = [np.min(x), np.max(x)]
xrd =  np.arange(xrd[0]-(xrd[1]-xrd[0])*0.2, xrd[1]+(xrd[1]-xrd[0])*0.2, (xr[1]-xr[0])/50)
bfit = np.linalg.lstsq(
    np.vstack((x, np.ones(len(x)))).T,
    np.matrix(y).T,
    rcond=None)
bfit = list(np.asarray(np.matmul(np.vstack((xrd, np.ones(len(xrd)))).T, bfit[0]).T).flatten())
'''for i in range(0, len(xrd)//2*2, 2):
    if(bfit[i] > yr[0] and bfit[i] < yr[1] and bfit[i+1] > yr[0] and bfit[i+1] < yr[1]):
        vs.append(draw.Line(mapRange(xr[0], xr[1], 0, bw, xrd[i  ])+(w-bw)/2, mapRange(yr[0], yr[1], bh, 0, bfit[i  ])+(h-bh)/2,
                            mapRange(xr[0], xr[1], 0, bw, xrd[i+1])+(w-bw)/2, mapRange(yr[0], yr[1], bh, 0, bfit[i+1])+(h-bh)/2,
                            stroke=post_stroke, fill='none', stroke_width=8, stroke_opacity=0.75))'''

xrd = [np.min(xp), np.max(xp)]
xrd =  np.arange(xrd[0]-(xrd[1]-xrd[0])*0.2, xrd[1]+(xrd[1]-xrd[0])*0.2, (xr[1]-xr[0])/50)
afit = np.linalg.lstsq(
    np.vstack((xp, np.ones(len(xp)))).T,
    np.matrix(yp).T,
    rcond=None)
afit = list(np.asarray(np.matmul(np.vstack((xrd, np.ones(len(xrd)))).T, afit[0]).T).flatten())
'''for i in range(0, len(xrd)//2*2, 2):
    if(afit[i] > yr[0] and afit[i] < yr[1] and afit[i+1] > yr[0] and afit[i+1] < yr[1]):
        vs.append(draw.Line(mapRange(xr[0], xr[1], 0, bw, xrd[i  ])+(w-bw)/2, mapRange(yr[0], yr[1], bh, 0, afit[i  ])+(h-bh)/2,
                            mapRange(xr[0], xr[1], 0, bw, xrd[i+1])+(w-bw)/2, mapRange(yr[0], yr[1], bh, 0, afit[i+1])+(h-bh)/2,
                            stroke='navy', fill='none', stroke_width=8, stroke_opacity=0.75))'''

xrd = [xr[0], xr[1]]
xrd = np.arange(xrd[0], xrd[1], (xrd[1]-xrd[0])/50)
xy = xrd.copy()
for i in range(0, len(xrd)//2*2, 2):
    if(xy[i] > yr[0] and xy[i] < yr[1] and xy[i+1] > yr[0] and xy[i+1] < yr[1]):
        vs.append(draw.Line(mapRange(xr[0], xr[1], 0, bw, xrd[i  ])+(w-bw)/2, mapRange(yr[0], yr[1], bh, 0, xy[i  ])+(h-bh)/2,
                            mapRange(xr[0], xr[1], 0, bw, xrd[i+1])+(w-bw)/2, mapRange(yr[0], yr[1], bh, 0, xy[i+1])+(h-bh)/2,
                            stroke=get_fgc(), fill='none', stroke_width=8, stroke_opacity=0.75))

arrow = draw.Marker(-0.002, -0.51, 0.9, 0.5, scale=4, orient='auto')
arrow.append(draw.Lines(-0.1, 0.5, -0.1, -0.5, 0.9, 0, fill=get_fgc(), close=True, opacity=0.5))

for i in range(len(x)):
    p0 = [mapRange(xr[0], xr[1], 0, bw, xp[i])+(w-bw)/2, mapRange(yr[0], yr[1], bh, 0, yp[i])+(h-bh)/2]
    p1 = [mapRange(xr[0], xr[1], 0, bw,  x[i])+(w-bw)/2, mapRange(yr[0], yr[1], bh, 0,  y[i])+(h-bh)/2]
    p0, p1 = np.array(p0), np.array(p1)
    p0 = p0+23*(p1-p0)/np.linalg.norm(p1-p0)
    p1 = p1-(25+30)*(p1-p0)/np.linalg.norm(p1-p0)
    vs.append(draw.Line(p0[0], p0[1], p1[0], p1[1], 
                        stroke=get_fgc(), stroke_width=10, stroke_opacity=0.5, fill='none', marker_end=arrow))

vs.append(draw.Use(scatter_svg(x, y, width=bw, height=bh, xr=xr, yr=yr, fill=post_fill, stroke=post_stroke, r=25), (w-bw)/2, (h-bh)/2))
vs.append(draw.Use(scatter_svg(xp, yp, width=bw, height=bh, xr=xr, yr=yr, fill='blue', stroke='navy', r=25), (w-bw)/2, (h-bh)/2))


yr=[0, 90]
y = list(100*np.divide(np.subtract(x, y), x))
yp = list(100*np.divide(np.subtract(xp, yp), xp))
de.append(draw.Use(trim_svg(xr, yr, width=bw, height=bh, yticks=8, ygs=1, title='relative improvement', xlabel='physics model RMSE (kcal/mol)', ylabel='relative improvement %')
    , (w-bw)/2, (h-bh)/2))


xrd = [np.min(x), np.max(x)]
xrd =  np.arange(xrd[0]-(xrd[1]-xrd[0])*0.2, xrd[1]+(xrd[1]-xrd[0])*0.2, (xr[1]-xr[0])/50)
bfit = np.linalg.lstsq(
    np.vstack((x, np.ones(len(x)))).T,
    np.matrix(y).T,
    rcond=None)
bfit = list(np.asarray(np.matmul(np.vstack((xrd, np.ones(len(xrd)))).T, bfit[0]).T).flatten())
'''for i in range(0, len(xrd)//2*2, 2):
    if(bfit[i] > yr[0] and bfit[i] < yr[1] and bfit[i+1] > yr[0] and bfit[i+1] < yr[1]):
        de.append(draw.Line(mapRange(xr[0], xr[1], 0, bw, xrd[i  ])+(w-bw)/2, mapRange(yr[0], yr[1], bh, 0, bfit[i  ])+(h-bh)/2,
                            mapRange(xr[0], xr[1], 0, bw, xrd[i+1])+(w-bw)/2, mapRange(yr[0], yr[1], bh, 0, bfit[i+1])+(h-bh)/2,
                            stroke=post_stroke, fill='none', stroke_width=8, stroke_opacity=0.75))'''

xrd = [np.min(xp), np.max(xp)]
xrd =  np.arange(xrd[0]-(xrd[1]-xrd[0])*0.2, xrd[1]+(xrd[1]-xrd[0])*0.2, (xr[1]-xr[0])/50)
afit = np.linalg.lstsq(
    np.vstack((xp, np.ones(len(xp)))).T,
    np.matrix(yp).T,
    rcond=None)
afit = list(np.asarray(np.matmul(np.vstack((xrd, np.ones(len(xrd)))).T, afit[0]).T).flatten())
'''for i in range(0, len(xrd)//2*2, 2):
    if(afit[i] > yr[0] and afit[i] < yr[1] and afit[i+1] > yr[0] and afit[i+1] < yr[1]):
        de.append(draw.Line(mapRange(xr[0], xr[1], 0, bw, xrd[i  ])+(w-bw)/2, mapRange(yr[0], yr[1], bh, 0, afit[i  ])+(h-bh)/2,
                            mapRange(xr[0], xr[1], 0, bw, xrd[i+1])+(w-bw)/2, mapRange(yr[0], yr[1], bh, 0, afit[i+1])+(h-bh)/2,
                            stroke='navy', fill='none', stroke_width=8, stroke_opacity=0.75))'''


for i in range(len(x)):
    p0 = [mapRange(xr[0], xr[1], 0, bw, xp[i])+(w-bw)/2, mapRange(yr[0], yr[1], bh, 0, yp[i])+(h-bh)/2]
    p1 = [mapRange(xr[0], xr[1], 0, bw,  x[i])+(w-bw)/2, mapRange(yr[0], yr[1], bh, 0,  y[i])+(h-bh)/2]
    p0, p1 = np.array(p0), np.array(p1)
    p0 = p0+23*(p1-p0)/np.linalg.norm(p1-p0)
    p1 = p1-(25+30)*(p1-p0)/np.linalg.norm(p1-p0)
    de.append(draw.Line(p0[0], p0[1], p1[0], p1[1], 
                        stroke=get_fgc(), stroke_width=10, stroke_opacity=0.5, fill='none', marker_end=arrow))

de.append(draw.Use(scatter_svg(x, y, width=bw, height=bh, xr=xr, yr=yr, fill=post_fill, stroke=post_stroke, r=25), (w-bw)/2, (h-bh)/2))
de.append(draw.Use(scatter_svg(xp, yp, width=bw, height=bh, xr=xr, yr=yr, fill='blue', stroke='navy', r=25), (w-bw)/2, (h-bh)/2))



vs_d.append(draw.Use(vs, 0, 0))
de_d.append(draw.Use(de, 0, 0))
vs_d.set_pixel_scale(2)
de_d.set_pixel_scale(2)
vs_d.save_svg('graphs/other/improvement1.svg')
de_d.save_svg('graphs/other/improvement2.svg')
vs_d.save_png('graphs/other/improvement1.png')
de_d.save_png('graphs/other/improvement2.png')

###############################################   Hyper 1   ###############################################

train, valid, diff, k = t1_results()

r = [min(np.min(valid), np.min(train)), max(np.max(valid), np.max(train))]
r[0] = np.floor(r[0])
r[1] = np.ceil(r[1])
dr = draw.Drawing(1450, 3600)

dr.append(draw.Rectangle(0, 0, 1300*2, 3600*2, fill=bgc))

gl = 3400#gauge length
gx = 1200
step = gl//10
for i in range(0, gl+step, step):
    kcal = mapRange(0, gl, r[0], r[1], i)
    dr.append(draw.Text("{:.1f}".format(kcal), 75, gx+100, (3600-gl)/2+i+30, fill = fgc, fill_opacity=0.75, font_family="Arial"))
step = gl//400
for i in range(0, gl, step):
    dr.append(draw.Rectangle(gx, (3600-gl)/2+i, 50, step+2, fill=colorRamp(mapRange(0, gl, 0, 1, i))))
sw = 5
dr.append(draw.Rectangle(gx+sw*0.5, (3600-gl)/2+sw*0.5, 50-sw, gl-2*sw+7, fill='None', fill_opacity=0, stroke=fgc, stroke_width=sw, stroke_opacity=0.5))
txt = draw.Text('RMSE - kcal/mol', 85, -3600*0.5, gx-15, text_anchor='middle', transform='rotate(-90)', fill=fgc, fill_opacity=0.75, font_family="Arial")
txtg = draw.Group()
txtg.append(txt)
dr.append(txtg)

dr.append(draw.Text('train'      , 120, 650,    0+150+20, fill=fgc, fill_opacity=0.75, font_family="Arial", text_anchor='middle', cutoff=0.3))
dr.append(draw.Text('validation' , 120, 650, 1000+350+20, fill=fgc, fill_opacity=0.75, font_family="Arial", text_anchor='middle', cutoff=0.3))
dr.append(draw.Text('valid-train' , 120, 650, 2000+550+20, fill=fgc, fill_opacity=0.75, font_family="Arial", text_anchor='middle', cutoff=0.3))


dr.append(draw.Use(hist2d_svg(train, width=800, height=800, dif=diff, r=r, k=k, name1='dl', name2='drop', final1=27, final2=0.4, cutoff=0.3), 250, 200))
dr.append(draw.Use(hist2d_svg(valid, width=800, height=800, dif=diff, r=r, k=k, name1='dl', name2='drop', final1=27, final2=0.4, cutoff=0.3), 250, 1400))
dr.append(draw.Use(hist2d_svg(diff , width=800, height=800, dif=diff, r=r, k=k, name1='dl', name2='drop', final1=27, final2=0.4, cutoff=0.3), 250, 2600))
dr.set_pixel_scale(2)
dr.save_svg("graphs/other/hypergraph1.svg")
dr.save_png("graphs/other/hypergraph1.png")

###############################################   Hyper 2   ###############################################

train, valid, diff, k = t2_results()

r = [min(np.min(valid), np.min(train)), max(np.max(valid), np.max(train))]
r[0] = np.floor(r[0])
r[1] = np.ceil(r[1]*10)/10
dr = draw.Drawing(1450, 3600)

dr.append(draw.Rectangle(0, 0, 1300*2, 3600*2, fill=bgc))

gl = 3400#gauge length
gx = 1200
step = gl//10
for i in range(0, gl+step, step):
    kcal = mapRange(0, gl, r[0], r[1], i)
    dr.append(draw.Text("{:.1f}".format(kcal), 75, gx+100, (3600-gl)/2+i+30, fill = fgc, fill_opacity=0.75, font_family="Arial"))
step = gl//400
for i in range(0, gl, step):
    dr.append(draw.Rectangle(gx, (3600-gl)/2+i, 50, step+2, fill=colorRamp(mapRange(0, gl, 0, 1, i))))
sw = 5
dr.append(draw.Rectangle(gx+sw*0.5, (3600-gl)/2+sw*0.5, 50-sw, gl-2*sw+7, fill='None', fill_opacity=0, stroke=fgc, stroke_width=sw, stroke_opacity=0.5))
txt = draw.Text('RMSE - kcal/mol', 85, -3600*0.5, gx-15, text_anchor='middle', transform='rotate(-90)', fill=fgc, fill_opacity=0.75, font_family="Arial")
txtg = draw.Group()
txtg.append(txt)
dr.append(txtg)

dr.append(draw.Text('train'      , 120, 650,    0+150+20, fill=fgc, fill_opacity=0.75, font_family="Arial", text_anchor='middle'))
dr.append(draw.Text('validation', 120, 650, 1000+350+20, fill=fgc, fill_opacity=0.75, font_family="Arial", text_anchor='middle'))
dr.append(draw.Text('valid-train', 120, 650, 2000+550+20, fill=fgc, fill_opacity=0.75, font_family="Arial", text_anchor='middle'))


dr.append(draw.Use(hist2d_svg(train, width=800, height=800, dif=diff, r=r, k=k, name1='c1', name2='c2', final1=53, final2=38, cutoff=0.3), 250, 200))
dr.append(draw.Use(hist2d_svg(valid, width=800, height=800, dif=diff, r=r, k=k, name1='c1', name2='c2', final1=53, final2=38, cutoff=0.3), 250, 1400))
dr.append(draw.Use(hist2d_svg(diff , width=800, height=800, dif=diff, r=r, k=k, name1='c1', name2='c2', final1=53, final2=38, cutoff=0.3), 250, 2600))
dr.set_pixel_scale(2)
dr.save_svg("graphs/other/hypergraph2.svg")
dr.save_png("graphs/other/hypergraph2.png")