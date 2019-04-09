#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 03:23:33 2019

@author: ssaumya7
"""
import scipy as sp
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from scipy import io

import time
import random

then = time.time() #Time before the operations start

Nmax = 120
gamma = 1
Delta = 0*gamma
kappa = 1*gamma
lamda = 1*gamma


pb_1n0 = 0.96
pb_2n1 = 0.04


U = 0.1*gamma
F0 = 0
F1 = 0
F2 = 50

T_pts = 480;tlist = np.linspace(0.0,48.0,T_pts)
#T_pts = 312;tlist = np.linspace(0.0,15.0,T_pts)


F0_F0 = Qobj( np.matrix([[1,0,0],[0,0,0],[0,0,0]]) )
F0tF0 = tensor( qeye(Nmax+1), F0_F0 )
F1_F1 = Qobj( np.matrix([[0,0,0],[0,1,0],[0,0,0]]) )
F1tF1 = tensor( qeye(Nmax+1), F1_F1 )
F2_F2 = Qobj( np.matrix([[0,0,0],[0,0,0],[0,0,1]]) )
F2tF2 = tensor( qeye(Nmax+1), F2_F2 )




FA = Qobj(np.matrix([[F0,0,0],[0,F1,0],[0,0,F2]]) ) 
Fop = tensor(qeye(Nmax+1), FA )
bdb = tensor(num(Nmax+1),qeye(3) )


Th_ph_dm_t = qload('Th_ph_p98_p02_L')
rho_t = Th_ph_dm_t.states

phi2 = np.zeros((T_pts,1))
expF = np.zeros((T_pts,1))
n_exs = np.zeros((T_pts,1))

chi2 = np.zeros((T_pts,1))
xi2 = np.zeros((T_pts,1))

for at in range(T_pts):
    Pg = kappa*pb_1n0*F1tF1*rho_t[at]
    phi2[at] = Pg.tr()
    
    
    PgA = kappa*F0tF0*rho_t[at]
    chi2[at] = PgA.tr()
    
    PgB = kappa*F2tF2*rho_t[at]
    xi2[at] = PgB.tr()
        
    
    
    Qg = Fop*rho_t[at]
    expF[at] = Qg.tr()
    
    Rg = bdb*rho_t[at]
    n_exs[at] = Rg.tr()

    
    
L_rho_F = rho_t[T_pts-1]
pro_n = np.zeros((Nmax+1,1))
    
#Pn = rho_F.diagonal()    

for an in range(Nmax+1):
    Ops= tensor(basis(Nmax+1, an)*basis(Nmax+1, an).trans(), qeye(3) )
    pro_n[an,0] = (Ops*L_rho_F).tr()

Pn = L_rho_F.diag()    

x = np.linspace(0,Nmax ,Nmax+1)

#markerline, stemlines, baseline = plt.stem(x, rho_F, '-.')
#plt.setp(baseline, color='r', linewidth=2)
#plt.show()

# Calculate the area with the trapezoidal rule
summ = 0; dtt = tlist[1] - tlist[0]
for at in range(1,T_pts-1):
    summ = summ + 2*phi2[at,0];

summ = summ + phi2[0,0] + phi2[T_pts-1,0]
AreA = summ/2*dtt;
print(AreA)

fig, ax = subplots()
ax.plot(tlist, n_exs);
ax.set_ylabel('<n>(t)');
ax.set_xlabel('Time');
show(fig)
fig.savefig('./A1_r_L')

fig, ax = subplots()
ax.plot(tlist, phi2);
ax.set_ylabel('|phi(t)|^2');
ax.set_xlabel('Time');
show(fig)
fig.savefig('./phi2_thermal_noise_L')




fig, ax = subplots()
ax.plot(tlist, expF);
ax.set_ylabel('<F>(t)');
ax.set_xlabel('Time');
show(fig)
fig.savefig('./A3_r_L')

fig, ax = subplots()
ax.plot(x, pro_n);
ax.set_ylabel('<n>(Steady state)');
ax.set_xlabel('boson number,n');
show(fig)
fig.savefig('./A4_r_L')

fig, ax = subplots()
ax.plot(tlist, chi2);
ax.set_ylabel('|chi(t)|^2');
ax.set_xlabel('Time');
show(fig)
fig.savefig('./Bc_L')

fig, ax = subplots()
ax.plot(tlist, xi2);
ax.set_ylabel('|xi(t)|^2');
ax.set_xlabel('Time');
show(fig)
fig.savefig('./Bx_L')


#print(Pn)

