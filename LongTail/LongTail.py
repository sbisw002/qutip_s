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


pb_1n0_1 = 0.98
pb_1n0_15 = 0.97
pb_1n0_2 = 0.96

pb_2n1 = 0.04


U = 0.1*gamma
F0 = 0
F1 = 0
F2 = 50

T_pts1 = 480;t1_list = np.linspace(0.0,48.0,T_pts1)
T_pts15 = 150;t15_list = np.linspace(0.0,15.0,T_pts15)
T_pts2 = 150;t2_list = np.linspace(0.0,15.0,T_pts2)


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
rho_t1 = Th_ph_dm_t.states

Th_ph_dm_t = qload('Th_ph_p96_p04_K')
rho_t2 = Th_ph_dm_t.states

Th_ph_dm_t = qload('Th_ph_p97_p03_M')
rho_t15 = Th_ph_dm_t.states


phi2_1 = np.zeros((T_pts1,1))
phi2_15 = np.zeros((T_pts15,1))
phi2_2 = np.zeros((T_pts2,1))

expF = np.zeros((T_pts1,1))
n_exs = np.zeros((T_pts1,1))

chi2 = np.zeros((T_pts1,1))
xi2 = np.zeros((T_pts1,1))

for at in range(T_pts1):
    Pg = kappa*pb_1n0_1*F1tF1*rho_t1[at]
    phi2_1[at] = Pg.tr()
    
    PgA = kappa*F0tF0*rho_t1[at]
    chi2[at] = PgA.tr()
    
    PgB = kappa*F2tF2*rho_t1[at]
    xi2[at] = PgB.tr()
        
    
    
    Qg = Fop*rho_t1[at]
    expF[at] = Qg.tr()
    
    Rg = bdb*rho_t1[at]
    n_exs[at] = Rg.tr()


for at in range(T_pts2):

    Pg = kappa*pb_1n0_2*F1tF1*rho_t2[at]
    phi2_2[at] = Pg.tr()

for at in range(T_pts15):

    Pg = kappa*pb_1n0_15*F1tF1*rho_t15[at]
    phi2_15[at] = Pg.tr()

    
    
L_rho_F1 = rho_t1[T_pts1-1];  L_rho_F2 = rho_t2[T_pts2-1];  L_rho_F15 = rho_t15[T_pts15-1]   
pro_n1 = np.zeros((Nmax+1,1))
pro_n2 = np.zeros((Nmax+1,1))
pro_n15 = np.zeros((Nmax+1,1))
    
#Pn = rho_F.diagonal()    

for an in range(Nmax+1):
    Ops= tensor(basis(Nmax+1, an)*basis(Nmax+1, an).trans(), qeye(3) )
    pro_n1[an,0] = (Ops*L_rho_F1).tr()
    pro_n2[an,0] = (Ops*L_rho_F2).tr()
    pro_n15[an,0] = (Ops*L_rho_F15).tr()
    
Pn1 = L_rho_F1.diag()    

x = np.linspace(0,Nmax ,Nmax+1)

#markerline, stemlines, baseline = plt.stem(x, rho_F, '-.')
#plt.setp(baseline, color='r', linewidth=2)
#plt.show()

# Calculate the area with the trapezoidal rule
summ = 0; dtt1 = t1_list[1] - t1_list[0]
for at in range(1,T_pts1-1):
    summ = summ + 2*phi2_1[at,0];

summ = summ + phi2_1[0,0] + phi2_1[T_pts1-1,0]
AreA1 = summ/2*dtt1;
print(AreA1)

fig, ax = subplots()
ax.plot(t1_list, n_exs);
ax.set_ylabel('<n>(t)');
ax.set_xlabel('Time');
show(fig)
fig.savefig('./A1_r_L')

fig, ax = subplots()
ax.plot(t1_list, phi2);
ax.set_ylabel('|phi(t)|^2');
ax.set_xlabel('Time');
show(fig)
fig.savefig('./phi2_thermal_noise_L')




fig, ax = subplots()
ax.plot(t1_list, expF);
ax.set_ylabel('<F>(t)');
ax.set_xlabel('Time');
show(fig)
fig.savefig('./A3_r_L')

fig, ax = subplots()
ax.plot(x, pro_n1);
ax.plot(x, pro_n2);
ax.set_ylabel('<n>(Steady state)');
ax.set_xlabel('boson number,n');
show(fig)
fig.savefig('./A4_r_L')



fig, ax = subplots()
ax.plot(t1_list, chi2);
ax.set_ylabel('|chi(t)|^2');
ax.set_xlabel('Time');
show(fig)
fig.savefig('./Bc_L')

fig, ax = subplots()
ax.plot(t1_list, xi2);
ax.set_ylabel('|xi(t)|^2');
ax.set_xlabel('Time');
show(fig)
fig.savefig('./Bx_L')


#print(Pn)
fig, ax = subplots()
#line1, = ax.plot(tlist, phi2);
line2, = ax.plot(t1_list, phi2_1);
line3, = ax.plot(t2_list, phi2_2);
ax.set_ylabel(r'$|\phi(t)|^2$');
ax.set_xlabel('Time,t');
ax.set_xlim(0, 15)
#ax.set_title(r'$p_{1 \rightarrow 0}=1$, Area under each of the curves = 1')
legend((line2, line3), (r'$p_{1 \rightarrow 0}=p_{1 \rightarrow 2}=0.98$', r'$p_{1 \rightarrow 0}=p_{1 \rightarrow 2}=0.96$'))
show(fig)
fig.savefig('./phi2_comparison_A.pdf')


fig, ax = subplots()
line1, = ax.plot(t1_list, phi2_1);
line2, = ax.plot(t15_list, phi2_15);
line3, = ax.plot(t2_list, phi2_2);
ax.set_ylabel(r'$|\phi(t)|^2$');
ax.set_xlabel('Time,t');
ax.set_xlim(0, 15)
#ax.set_title(r'$p_{1 \rightarrow 0}=1$, Area under each of the curves = 1')
legend((line1, line2, line3), (r'$p_{1 \rightarrow 0}=p_{1 \rightarrow 2}=0.98$', r'$p_{1 \rightarrow 0}=p_{1 \rightarrow 2}=0.97$', r'$p_{1 \rightarrow 0}=p_{1 \rightarrow 2}=0.96$'))
show(fig)
fig.savefig('./phi2_comparison_B.pdf')

