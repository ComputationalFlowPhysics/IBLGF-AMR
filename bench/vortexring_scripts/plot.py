#!/usr/bin/env python

import os
import math
import matplotlib
import copy
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import matplotlib.ticker as tickr
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from matplotlib import rc
import matplotlib.ticker as ticker
import matplotlib.ticker as mticker
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator

from matplotlib.ticker import AutoMinorLocator
from matplotlib import rc

from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)

#from plotit import *
from operator import itemgetter


def figsize(scale):
	#fig_width_pt  = 384.0                            # Get this from LaTeX using \the\textwidth
	fig_width_pt  = 510.0                             # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                         # Convert pt to inch
	golden_mean   = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
	fig_width     = fig_width_pt*inches_per_pt*scale  # width in inches
	fig_height    = fig_width*golden_mean             # height in inches
	fig_size      = [fig_width,fig_height]
	return fig_size

pgf_with_latex = {                          # setup matplotlib to use latex for output
	"pgf.texsystem": "pdflatex",           # change this if using xetex or lautex
	"text.usetex": True,                   # use LaTeX to write all text
	"font.family": "serif",
	"font.serif": [],                      # blank entries should cause plots to inherit fonts from the document
	"font.sans-serif": [],
	"font.monospace": [],
	"axes.labelsize": 18,                  # LaTeX default is 10pt font.
	"font.size": 18,
	"legend.fontsize": 12,                 # Make the legend/label fonts a little smaller
	"xtick.labelsize": 18,
	"ytick.labelsize": 18,
	"lines.marker": "None",                # the default marker
	"lines.markeredgewidth": 1,            # the line width around the marker symbol
	"lines.markersize": 8,                 # markersize, in points
	"figure.figsize": figsize(0.9),        # default fig size of 0.9 textwidth
	"pgf.preamble": [
		r"\usepackage[utf8x]{inputenc}",   # use utf8 fonts becasue your computer can handle it :)
		r"\usepackage[T1]{fontenc}",       # plots will be generated using this preamble
	]
}
matplotlib.rcParams.update(pgf_with_latex)

def newfig(width):
	fig = plt.figure(figsize=figsize(width))
	ax = fig.add_subplot(111)
	return fig, ax
	

 
def savefig(filename):
	plt.savefig('{}.eps'.format(filename),bbox_inches='tight')

def get_data(file_name, nHeaderLines=0):
	return np.loadtxt(file_name ,skiprows=0)

def get_data_whiteSpace(file_name, nHeaderLines=0):
	return np.loadtxt(file_name ,skiprows=0,delimiter=' ')

def get_data_kommaDel(file_name, nHeaderLines=0):
	return np.loadtxt(file_name ,skiprows=0,delimiter=',')



def main():
 
    dash_style_1= []
    dash_style_2= [6, 4] #Dashed
    dash_style_3= [8, 2, 1, 2] #line-dot
    dash_style_4= [8, 2, 1, 2, 1, 2] #line-doubleShortDash
    dash_style_5= [8, 2, 1, 2, 1, 2, 1, 2] #triple-dashed line
    dash_style_6= [2, 2] #Dotted
    all_styles=[dash_style_1, dash_style_2, dash_style_3, dash_style_4,
    dash_style_5,dash_style_6];
    colors=["red"]
    line_width=1
    markersize=8

    fig, ax = newfig(1.0)
    lines=[]
    labels=[]

    filename = "error.txt"
    data_all=get_data(filename)
    levels=data_all[:,0]
    L2=data_all[:,1]
    LInf=data_all[:,2]
    print("level ", levels)


    #L2
    #LInf
    dx_base=0.0125
    dx=np.zeros(len(levels))
    for i in range(len(levels)):
        dx[i]=dx_base/(2**levels[i])


    dx=dx[::-1]
    L2=L2[::-1]
    LInf=LInf[::-1]

    line0, =ax.plot(dx,L2, 'ro',lw=line_width)
    line0.set_dashes(all_styles[1])
    line0.set_label(r'$L_2$ ')

    line0, =ax.plot(dx,LInf, 'bo',lw=line_width)
    line0.set_dashes(all_styles[2])
    line0.set_label(r'$L_\infty$ ')

    second_order =1.5e-1* (dx)**2
    line0, =ax.plot(dx,second_order, 'k',lw=line_width)
    #line0.set_dashes(all_styles[5])
    line0.set_label(r'2nd order')


    ax.set_xscale('log')
    ax.set_yscale('log')
    #ax.set_xticks([])
    #yticks=[0.1]
    #plt.xticks(yticks)

    lines.append(line0)
    #ax.legend(mode="expand", ncol=3)
    ax.legend(ncol=1)
    ax.set_ylabel(r'Error')
    ax.set_xlabel(r'$\Delta x$')
    outname="convergence_amr"
    print("Writing to file: ",outname)
    savefig(outname)


main()


