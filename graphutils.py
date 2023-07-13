

import drawsvg as draw
from utils import *


bgc = '#ffffff'
fgc = '#000000'
#bgc, fgc = fgc, bgc
stroke_scale = 100
sw = max(1000, 1000)/stroke_scale

t_scale = {'title' : 120, 'label' : 95, 'numbers' : 75}

sa = 0.75 # stroke alpha
ta = 0.75 # text alpha

def get_fgc():
    return fgc

def get_bgc():
    return bgc

def colorRamp(a):
    return rgb2hex(hsv2rgb(((1-a)*0.75, 0.60, 0.75)))

def rgb2hex(rgb):
    r = min(255, max(0, int(rgb[0])))
    g = min(255, max(0, int(rgb[1])))
    b = min(255, max(0, int(rgb[2])))
    h = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
    return "#" + h[r//16] + h[r%16] + h[g//16] + h[g%16] + h[b//16] + h[b%16]

def hsv2rgb(hsv):
    h = hsv[0]%1
    s = min(1, max(0, hsv[1]))
    v = min(1, max(0, hsv[2]))
    c = v*s
    x = c*(1-abs((h*6)%2-1))
    m = v-c
    if(h<1/6):
        p = (c, x, 0)
    elif(h<2/6):
        p = (x, c, 0)
    elif(h<3/6):
        p = (0, c, x)
    elif(h<4/6):
        p = (0, x, c)
    elif(h<5/6):
        p = (x, 0, c)
    else:
        p = (c, 0, x)
    return((p[0]+m)*255, (p[1]+m)*255, (p[2]+m)*255)

def hist_svg(x, y, width = 1000, height = 1000, xr = None, yr = None, stroke = 'black', fill = 'red', alpha=0.75, s_scale=1):
    if(type(xr) == type(None)):
        xr = [np.min(x), np.max(x)]
    if(type(yr) == type(None)):
        yr = [np.min(y), np.max(y)]

    d = draw.Group()
    dx = x[1]-x[0]
    xr[1] += dx
    
    pf = draw.Path(stroke_width=sw*s_scale, stroke_opacity=sa, stroke='none', fill=fill, opacity=alpha)
    po = draw.Path(stroke_width=sw*s_scale, stroke_opacity=sa*alpha, stroke=stroke, fill='none')
    pf.M(mapRange(xr[0], xr[1], 0, width, 0), mapRange(yr[0], yr[1], height, 0, 0))
    po.M(mapRange(xr[0], xr[1], 0, width, 0)+sw/2*s_scale, mapRange(yr[0], yr[1], height, 0, 0))
    for i in range(len(x)):
        if(i==len(x)-1 or y[i+1]<y[i]):
            xo = -sw/2*s_scale
        else:
            xo = sw/2*s_scale
        if(i==len(x)-1 or y[i] <= 0):
            vo = 0
        else:
            vo = 1

        pf.V(mapRange(yr[0], yr[1], height, 0, y[i]))
        po.V(mapRange(yr[0], yr[1], height, 0, y[i])+sw/2*s_scale*vo)
        if(y[i]>0):
            pf.H(mapRange(xr[0], xr[1], 0, width, x[i]+dx))
            po.H(mapRange(xr[0], xr[1], 0, width, x[i]+dx)+xo)
        else:
            pf.M(mapRange(xr[0], xr[1], 0, width, x[i]+dx), height)
            po.M(mapRange(xr[0], xr[1], 0, width, x[i]+dx)+xo, height)

    pf.V(height-mapRange(yr[0], yr[1], 0, height, 0))
    po.V(height-mapRange(yr[0], yr[1], 0, height, 0))
    d.append(pf)
    d.append(po)
    return d


def scatter_svg(x, y, width = 1000, height = 1000, xr = None, yr = None, stroke='black', fill='red', r=15, s_scale=None, alpha=1):
    if(type(xr) == type(None)):
        xr = [np.min(x), np.max(x)]
    if(type(yr) == type(None)):
        yr = [np.min(y), np.max(y)]
    if(type(s_scale) == type(None)):
        s_scale = r/2/sw
    d = draw.Group()
    #d.append(draw.Rectangle(0, 0, width, height, fill=bgc, stroke='none'))

    for pt in np.vstack((x, y)).T:
        d.append(draw.Circle(mapRange(xr[0], xr[1], 0, width, pt[0]), mapRange(yr[0], yr[1], height, 0, pt[1])
                    , r, fill=fill, opacity=alpha, stroke='none'))

        d.append(draw.Circle(mapRange(xr[0], xr[1], 0, width, pt[0]), mapRange(yr[0], yr[1], height, 0, pt[1])
                    , r-sw/2*s_scale, fill='none', stroke=stroke, stroke_width=sw*s_scale, stroke_opacity=sa*alpha))


    return d

def plot_svg(x, y, width = 1000, height = 1000, xr = None, yr = None, stroke = 'black', fill = 'red', s_scale = 1, alpha = sa):
    if(type(xr) == type(None)):
        xr = [np.min(x), np.max(x)]
    if(type(yr) == type(None)):
        yr = [np.min(y), np.max(y)]

    d = draw.Group()
    #d.append(draw.Rectangle(0, 0, width, height, fill=bgc, stroke='none'))

    p = draw.Path(stroke_width=sw*s_scale, stroke_opacity=alpha, stroke=stroke, fill='none')
    p.M(mapRange(xr[0], xr[1], 0, width, x.pop(0)), mapRange(yr[0], yr[1], height, 0, y.pop(0)))
    for pt in np.vstack((x, y)).T:
        p.L(mapRange(xr[0], xr[1], 0, width, pt[0]), mapRange(yr[0], yr[1], height, 0, pt[1]))
    d.append(p)
    return d


def trim_svg(xr, yr, width=1000, height=1000, xticks=3, yticks=3, s_scale=1, title=None, xlabel=None, ylabel=None, ygs=2, xgs=2):

    scales = [0.01, 0.1, 0.25, 0.5, 1, 2, 5, 10, 100, 250, 500]
    prec   = [2   , 1  , 2   , 1  , 0, 0, 0,  0,   0,   0,   0]

    xi = scales[np.argmin(np.abs(np.divide(np.ptp(xr), scales)-xticks))]
    yi = scales[np.argmin(np.abs(np.divide(np.ptp(yr), scales)-yticks))]

    xp = prec[np.where(np.array(scales) == xi)[0][0]]
    yp = prec[np.where(np.array(scales) == yi)[0][0]]

    d = draw.Group()
    tick_length = 30

    d.append(draw.Line(-sw/2*s_scale, 0, -sw/2*s_scale, height+sw*s_scale, fill='none', stroke=fgc, stroke_width=sw*s_scale, stroke_opacity=sa))
    d.append(draw.Line(-sw*s_scale, height+sw/2*s_scale, width, height+sw/2*s_scale, fill='none', stroke=fgc, stroke_width=sw*s_scale, stroke_opacity=sa))

    for i in np.arange(np.min(xr)//xi*xi+xi, np.max(xr), xi):
        d.append(draw.Line(mapRange(xr[0], xr[1], 0, width, i), height+sw*s_scale, mapRange(xr[0], xr[1], 0, width, i), height+tick_length,
            fill='none', stroke=fgc, stroke_width=sw*s_scale, stroke_opacity=sa))
        d.append(draw.Text(("{:."+str(xp)+"f}").format(i), t_scale['numbers'], mapRange(xr[0], xr[1], 0, width, i), height+tick_length+60,
            text_anchor='middle', fill=fgc, fill_opacity=ta, font_family="Arial"))
    
    xi = xi/xgs
    for i in np.arange(np.min(xr)//xi*xi+xi, np.max(xr), xi):
        d.append(draw.Line(mapRange(xr[0], xr[1], 0, width, i), 0, mapRange(xr[0], xr[1], 0, width, i), height,
            fill='none',stroke=rgb2hex((30, 30, 70)), stroke_width=sw*0.75*s_scale, stroke_opacity=sa/3))

    for i in np.arange(np.min(yr)//yi*yi+yi, np.max(yr), yi):
        d.append(draw.Line(-sw*s_scale, mapRange(yr[0], yr[1], height, 0, i), -tick_length, mapRange(yr[0], yr[1], height, 0, i),
            ill='none', stroke=fgc, stroke_width=sw*s_scale, stroke_opacity=sa))
        d.append(draw.Text(("{:."+str(yp)+"f}").format(i), t_scale['numbers'], -tick_length-10, 23+mapRange(yr[0], yr[1], height, 0, i),
            text_anchor='end', fill=fgc, fill_opacity=ta, font_family="Arial"))
    
    yi = yi/ygs
    for i in np.arange(np.min(yr)//yi*yi+yi, np.max(yr), yi):
        d.append(draw.Line(0, mapRange(yr[0], yr[1], height, 0, i), width, mapRange(yr[0], yr[1], height, 0, i),
            fill='none',stroke=rgb2hex((30, 30, 70)), stroke_width=sw*0.75*s_scale, stroke_opacity=sa/3))

    d.append(draw.Text(title, t_scale['title'], width/2, -75, 
        text_anchor='middle', fill=fgc, fill_opacity=ta, font_family="Arial"))

    
    if(type(ylabel) != type(None)):
        txtg = draw.Group()
        txt = draw.Text(ylabel, t_scale['label'], -height/2, -160, transform='rotate(-90)', text_anchor='middle', fill=fgc, fill_opacity=ta, font_family='Arial')
        #txt = draw.Text(ylabel, t_scale['label'], -height/2, -140)
        txtg.append(txt)
        d.append(txtg)
    if(type(xlabel) != type(None)):
        d.append(draw.Text(xlabel, t_scale['label'], width/2, height+164, text_anchor='middle', fill=fgc, fill_opacity=ta, font_family="Arial"))

    return d



