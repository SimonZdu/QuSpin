#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 20:57:11 2019

@author: simon.d
"""
from __future__ import print_function, division
import sys,os
qspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,qspin_path)
#####################################################################
#                           2d transverse ising                    #
#In this script we set up a 2d transverse ising model 
#H=sum_{i j in sites} J*s_i^z*s_j^z + h*s_x. We diagonalize H and 
#observe the magnetization of each energy eigenstate.
#This is a stable version
                   
#####################################################################
from quspin.operators import hamiltonian # operators
from quspin.basis import spin_basis_general # spin basis constructor
import numpy as np # general math functions
import matplotlib.pyplot as plt # plotting library
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
startTime = datetime.now()
#
###### define model parameters ######
J=-32 # spin=spin interaction
g=48 # x direction magnetic field strength
h=0.015625# z direction mag field strength
#
Lx, Ly = 4, 4 # linear dimension of spin 1/2 2d lattice 
# computation time is ~ 2^(3*(Lx*Ly))
# Lx*Ly=12 takes 2.5 sec
# Lx*Ly=16 takes 24 min
N_2d = Lx*Ly # number of sites for spin
#
###### setting up user-defined symmetry transformations for 2d lattice ######
s = np.arange(N_2d) # sites [0,1,2,....]
x = s%Lx # x positions for sites
y = s//Lx # y positions for sites
T_x = (x+1)%Lx + Lx*y # translation along x-direction
T_y = x +Lx*((y+1)%Ly) # translation along y-direction
P_x = x + Lx*(Ly-y-1) # reflection about x-axis
P_y = (Lx-x-1) + Lx*y # reflection about y-axis
Z   = -(s+1) # spin inversion
#
Jzz=[[J,i,T_x[i]] for i in range(N_2d)]+[[-1.0,i,T_y[i]] for i in range(N_2d)]
 
gx =[[g,i] for i in range(N_2d)]
gz=[[h,i] for i in range(N_2d)]
#
static=[["zz",Jzz],["x",gx],["z",gz]]
static_M=[["z",[[1,i] for i in range(N_2d)]]]
#static_M2=[]
#
E=list()
M_expt=list()
M_var=list()
for kx in range(Lx): # all x translation eigenvalues
    for ky in range(Ly): # all y translation eigenvalues
        basis_2d = spin_basis_general(N_2d,kxblock=(T_x,kx),kyblock=(T_y,ky))
        #basis that span eigenspace of translations with eigenvalue (kx, ky) 
        H=hamiltonian(static,[],basis=basis_2d,dtype=np.complex128)
        M=hamiltonian(static_M,[],basis=basis_2d,dtype=np.complex128) #total magnetization op.
        #M2=hamiltonian(static_M,[],basis=basis_2d,dtype=np.complex128)
        # diagonalise H
        E_kblock,V_kblock=H.eigh()
        M_expt_kblock = M.expt_value(V_kblock,time=0,check=False,enforce_pure=True)
        E.extend(np.ndarray.tolist(E_kblock)) # collect eigenenergies
        M_expt.extend(np.ndarray.tolist(np.real(M_expt_kblock)))#collect magnetization expt. of H eigenstates

# the plot
pltx = M_expt
plty = E

fig, axScatter = plt.subplots(figsize=(8, 8))

# the scatter plot:
axScatter.scatter(pltx, plty)
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

xmax = np.max(np.abs(pltx))
ymax = np.max(np.abs(plty))
xbinwidth = xmax/16
ybinwidth = ymax/8 
xlim = (int(xmax/xbinwidth) + 1)*xbinwidth
ylim = (int(ymax/ybinwidth) + 1)*ybinwidth

xbins = np.arange(-xlim, xlim + xbinwidth, xbinwidth)
filtered_M = [M_expt[i] for i in range(len(E)) if E[i]<(-200)]
axHistx.hist(filtered_M, bins=xbins)
#axHistx.ylabel('DOS',fontsize=16)
ybins = np.arange(-ylim, ylim + ybinwidth, ybinwidth)
axHisty.hist(plty, bins=ybins, orientation='horizontal')
#axHisty.ylabel('DOS',fontsize=16)
# the xaxis of axHistx and yaxis of axHisty are shared with axScatter,
# thus there is no need to manually adjust the xlim and ylim of these
# axis.

plt.draw()
plt.show()
fig_name_str="J_g_Lx_Ly"+"_"+np.str(J)+"_"+np.str(g)+"_"+np.str(Lx)+"_"+np.str(Ly)
plt.savefig('2d_transverse_ising'+fig_name_str+'.pdf', bbox_inches='tight')


print ('run time,', datetime.now()-startTime)