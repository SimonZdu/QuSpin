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
#observe the m^2 of all energy eigenstate.

#####################################################################
from quspin.operators import hamiltonian # operators
from quspin.basis import spin_basis_general # spin basis constructor
import numpy as np # general math functions
import matplotlib.pyplot as plt # plotting library
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from quspin.tools import plotmvse
from datetime import datetime
startTime = datetime.now()
#
###### define model parameters ######
J=-32 # spin=spin interaction
g=32# x direction magnetic field strength
h=0.03125 #z direction mag field strength
Lx, Ly = 3, 3 # linear dimension of spin 1/2 2d lattice 
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
E_den=list()
M_expt=list()
M_var=list()
m_var=list()
for kx in range(Lx): # all x translation eigenvalues
    for ky in range(Ly): # all y translation eigenvalues
        basis_2d = spin_basis_general(N_2d,kxblock=(T_x,kx),kyblock=(T_y,ky))
        #basis that span eigenspace of translations with eigenvalue (kx, ky) 
        H=hamiltonian(static,[],basis=basis_2d,dtype=np.complex128,check_symm=False, check_herm=False, check_pcon=False)
        M=hamiltonian(static_M,[],basis=basis_2d,dtype=np.complex128,check_symm=False, check_herm=False, check_pcon=False) #total magnetization op.
        #M2=hamiltonian(static_M,[],basis=basis_2d,dtype=np.complex128)
        # diagonalise H
        E_kblock,V_kblock=H.eigh()
        m_var_kblock = M.quant_fluct(V_kblock,time=0,check=False,enforce_pure=True)/np.exp2(N_2d)
        E_den.extend(np.ndarray.tolist(E_kblock/np.exp2(N_2d))) # collect eigenenergies
        m_var.extend(np.ndarray.tolist(np.real(m_var_kblock)))#collect m_var of H eigenstates


# the plot

fig, axScatter = plt.subplots(figsize=(8, 8))

# the scatter plot:
axScatter.scatter(m_var, E_den)
plt.xlabel('m_var',fontsize=16)
plt.ylabel('energy',fontsize=16)

# create new axes on the right and on the top of the current axes
# The first argument of the new_vertical(new_horizontal) method is
# the height (width) of the axes to be created in inches.
divider = make_axes_locatable(axScatter)
axHistx = divider.append_axes("top", 1.2, pad=0.1, sharex=axScatter)
#axRF = divider.append_axes("top", 1.2, pad=0.2, sharex=axScatter)
axHisty = divider.append_axes("right", 1.2, pad=0.1, sharey=axScatter)

# make some labels invisible
axHistx.xaxis.set_tick_params(labelbottom=False)
axHisty.yaxis.set_tick_params(labelleft=False)

#code: now determine nice limits by hand:

xmax = np.max(np.abs(m_var))
ymax = np.max(np.abs(E_den))
xbinwidth = xmax/32
ybinwidth = ymax/32 
xlim = (int(xmax/xbinwidth) + 1)*xbinwidth
ylim = (int(ymax/ybinwidth) + 1)*ybinwidth
xbins = np.arange(0, xlim + xbinwidth, xbinwidth)
axHistx.hist(m_var, bins=xbins)
#RateFun = [M_expt[i]*np.exp() for i in range(len(E)) if E[i]<(-200)]
#axRF.plot(RateFun)# TBD!
#physics: The histogram should include Boltzmann factor!
# exp(- beta E' +hz M - T S(E',M,N))
ybins = np.arange(-ylim, ylim + ybinwidth, ybinwidth)
axHisty.hist(E_den, bins=ybins, orientation='horizontal')
# axHisty.ylabel('DOS',fontsize=16)
# the xaxis of axHistx and yaxis of axHisty are shared with axScatter,
# thus there is no need to manually adjust the xlim and ylim of these
# axis.

plt.draw()
plt.show()
fig_name_str="J_g_Lx_Ly"+"_"+np.str(J)+"_"+np.str(g)+"_"+np.str(Lx)+"_"+np.str(Ly)

print ('run time,', datetime.now()-startTime)