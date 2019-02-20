#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 13:13:49 2019

@author: simon.d
"""
#####################################################################
#specialized plot function for 2d_transverse_ising.py
#plot energy versus magnetization of each eigenstate as a scattered dot
#then count the number of states of certain magnetization/ energy and 
#plot the histogram                    
#####################################################################
import numpy as np # general math functions
import matplotlib.pyplot as plt # plotting library
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
	from itertools import izip as zip
except ImportError:
	pass

try:
	from functools import reduce as reduce
except ImportError:
	pass

import warnings


__all__ = ["plotmvse"]

def plotmvse(M,E):
# the plot

fig, axScatter = plt.subplots(figsize=(8, 8))

# the scatter plot:
axScatter.scatter(M, E)
plt.xlabel('total mag',fontsize=16)
plt.ylabel('energy',fontsize=16)
#axScatter.set_aspect(1.)

# create new axes on the right and on the top of the current axes
# The first argument of the new_vertical(new_horizontal) method is
# the height (width) of the axes to be created in inches.
divider = make_axes_locatable(axScatter)
axHistx = divider.append_axes("top", 1.2, pad=0.1, sharex=axScatter)
axHisty = divider.append_axes("right", 1.2, pad=0.1, sharey=axScatter)

# make some labels invisible
axHistx.xaxis.set_tick_params(labelbottom=False)
axHisty.yaxis.set_tick_params(labelleft=False)

# now determine nice limits by hand:

xmax = np.max(np.abs(M))
ymax = np.max(np.abs(E))
xbinwidth = xmax/32
ybinwidth = ymax/32 
xlim = (int(xmax/xbinwidth) + 1)*xbinwidth
ylim = (int(ymax/ybinwidth) + 1)*ybinwidth

xbins = np.arange(-xlim, xlim + xbinwidth, xbinwidth)
axHistx.hist(M, bins=xbins)
#axHistx.ylabel('DOS',fontsize=16)
ybins = np.arange(-ylim, ylim + ybinwidth, ybinwidth)
axHisty.hist(E, bins=ybins, orientation='horizontal')
#axHisty.ylabel('DOS',fontsize=16)

plt.draw()
plt.show()

#
