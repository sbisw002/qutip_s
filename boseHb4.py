#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 19:06:43 2019

@author: ssaumya7
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 22:09:47 2019

@author: ssaumya7
"""
import scipy as sp
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from scipy import io

Nmax = 130
gamma = 1
Delta = 0*gamma
kappa = 1*gamma
U = 0.1*gamma
F0 = 0
F1 = 50
Sgm = 55
t0 = 250
psi0 = tensor(basis(Nmax+1, 0), basis(2,0))
#print(psi0)
#b = tensor(destroy(Nmax+1), qeye(1))
b = tensor(destroy(Nmax+1), qeye(2) )
F1F0 = tensor( qeye(Nmax+1), Qobj( np.matrix([[0,1],[0,0]]) ) )
#print(b)

bdb = tensor(num(Nmax+1),qeye(2) )
#print(bdb)


bNp = tensor(basis(Nmax+1,Nmax)*basis(Nmax+1,Nmax).dag(), qeye(2)  )
b0p = tensor(basis(Nmax+1,0)*basis(Nmax+1,0).dag(), qeye(2) )



b_ops = []  # Build collapse operators
b_ops.append(np.sqrt(gamma) * b)
b_ops.append(np.sqrt(kappa) * F1F0)
FA = Qobj(np.matrix([[F0,0],[0,F1]]) ) 

Fop = tensor(qeye(Nmax+1), FA )
#print(FA)

H0 = tensor( Qobj(-Delta* num(Nmax+1) + U*( num(Nmax+1) ** 2 - num(Nmax+1)  )     ),qeye(2)  )  
H1 = tensor( ( destroy(Nmax+1) + create(Nmax+1) ), FA )  # time-dependent term
H2 = tensor( qeye(Nmax+1),sigmax() )
#H2 = tensor( qeye(Nmax+1),create(2) )

t = np.linspace(0.0, 500, 4566) # Define time vector
#t = np.append(np.linspace(-0.2, -0.1 - (-0.1+0.2)/86400, 86400), np.linspace(-0.1, 0.2, 7645640))


#print(H0)
#print(H1)
#print(H2)

def H2_coeff(tt, args=None):
    return 1*np.sin((15)*tt)*np.sqrt(kappa)*np.exp(-( (tt-t0) /np.sqrt(2)/Sgm  ) ** 2)/np.sqrt(2*np.pi*Sgm*Sgm)
#    return 0



N = 10

wc = 5.0 * 2 * np.pi
w1 = 3.0 * 2 * np.pi
w2 = 2.0 * 2 * np.pi

g1 = 0.01 * 2 * np.pi
g2 = 0.0125 * 2 * np.pi

T0_1 = 3
T0_2 = 4
width = 0.5
T_gate_1 = 2
T_gate_2 = 16

def step_t(w1, w2, t0, width, t):
    """
    Step function that goes from w1 to w2 at time t0
    as a function of t. 
    """
    return w1 + (w2 - w1) * (t > t0)


fig, axes = plt.subplots(1, 1, figsize=(8,2))
axes.plot(t, [step_t(0.5, 1.5, 2, 0.0, t) for t in t], 'k')
axes.set_ylim(0, 2)
fig.tight_layout()

def wc_t(t, args=None):
    return wc

#def w1_t(t, args=None):
#    return w1 + step_t(0.0, wc-w1, T0_1, width, t) - step_t(0.0, wc-w1, T0_1+T_gate_1, width, t)

#def w2_t(t, args=None):
#    return w2 + step_t(0.0, wc-w2, T0_2, width, t) - step_t(0.0, wc-w2, T0_2+T_gate_2, width, t)

def w1_t(t, args=None):
#    return step_t(0.0, 10*wc-w1, T0_1, width, t) - step_t(0.0, wc-w1, T0_1+T_gate_1, width, t)
    return H2_coeff(t)

def w2_t(t, args=None):
    return step_t(0.0, 20*wc-w2, T0_2, width, t) - step_t(0.0, 20*wc-w2, T0_2+T_gate_2, width, t)


fig, axes = plt.subplots(1, 1, figsize=(8,2))
axes.plot(t, [w1_t(t) for t in t], 'k')
#axes.set_ylim(0, 2)
fig.tight_layout()
fig, axes = plt.subplots(1, 1, figsize=(8,2))
axes.plot(t, [w2_t(t) for t in t], 'k')
#axes.set_ylim(0, 2)
fig.tight_layout()


#H_t = [[Hc, wc_t], [H1, w1_t], [H2, w2_t], Hc1+Hc2]


H01 = H0 + H1    
#H = [H01,[H2,H2_coeff]]
#H = [[H2, wc_t], [H2, w1_t], H01]
#H = [[H2, w1_t],[H2, w2_t], H01]

#H = [[H2, w1_t],H01]
H = [[H2, w1_t],H01]


#print(H)

Fs = H2_coeff(t,0)

fig, ax = subplots()
ax.plot(t, Fs);
ax.set_xlabel('Time');
ax.set_ylabel('F');
ax.legend(("F"));
show()



#print(H0)
#print(H1)
#print(psi0)




output = mesolve(H, psi0, t, b_ops, [bdb, b0p, bNp, Fop])

fig, ax = subplots()
ax.plot(output.times, output.expect[0]);
ax.plot(output.times, output.expect[1]);
ax.plot(output.times, output.expect[2]);
ax.plot(output.times, output.expect[3]);
ax.set_xlabel('Time');
ax.set_ylabel('Expectation values');
ax.legend(("es","0_occ", "Nmax_occ","<F>"));
show()


fig, ax = subplots()
ax.plot(output.times, output.expect[3]);
ax.set_xlabel('Time');
ax.set_ylabel('<F>');
ax.legend(("exp(F)"));
show()


fig, ax = subplots()
ax.plot(output.times, output.expect[2]);
ax.set_xlabel('Time');
ax.set_ylabel('rho_NmNm');
ax.legend(("Nmax_occ"));
show()



fig, ax = subplots()
ax.plot(output.times, output.expect[1]);
ax.set_xlabel('Time');
ax.set_ylabel('rho_00');
ax.legend(("0_occ"));
show()


fig, ax = subplots()
ax.plot( output.times,output.expect[0] );
ax.set_xlabel('t');
ax.set_ylabel('exc');
ax.legend(("excitations vs t"));
show()


fig, ax = subplots()
ax.plot( Fs,output.expect[0] );
ax.set_xlabel('F');
ax.set_ylabel('n');
ax.legend(("F vs n"));
show()


sp.io.savemat('./Dat_k15.mat', mdict={'Fs': Fs,'t': t,'bdb': output.expect[0],'b0p': output.expect[1],'bNp': output.expect[2],'exp_F': output.expect[3]})

