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

from matplotlib.pyplot import *


Nmax = 60
gamma = 1
Delta = 1*gamma
kappa = 1*gamma
U = 0.1*gamma
F0 = 0
F1 = 50
Sgm = 0.00125
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



mA = np.diag(np.ones(Nmax),1)
n1A = np.diag(np.sqrt(np.arange(1,Nmax+1)),1) # * F
n2A = np.diag(np.sqrt(np.arange(1,Nmax+1)),-1) # * F

oA = np.diag(  np.arange(Nmax+1 )  ) # * (-Delta)
pA = np.diag( np.arange(Nmax+1) * np.arange(Nmax+1) -  np.arange(Nmax+1) 
  ) # * U
#print(mA)
#print(nA)
#print(oA)
#print(pA)
#print(psi0)


b_ops = []  # Build collapse operators
b_ops.append(np.sqrt(gamma) * b)
b_ops.append(np.sqrt(kappa) * F1F0)
FA = Qobj(np.matrix([[F0,0],[0,F1]]) )

Fop = tensor(qeye(Nmax+1), FA )
#print(FA)

H0 = tensor( Qobj(-Delta*oA + U*pA),qeye(2)  )
H1 = tensor( Qobj( n1A + n2A ), FA )  # time-dependent term
H2 = tensor( qeye(Nmax+1),sigmax() )

#print(H0)
#print(H1)
#print(H2)

def H2_coeff(t, args):
     return np.sqrt(kappa)*np.exp(-(  (t-0.2) /np.sqrt(2)/Sgm  ) ** 2)/np.sqrt(2*np.pi*Sgm*Sgm)
#    return 0


H01 = H0 + H1
H = [H01,[H2,H2_coeff]]

#print(H)

t = np.linspace(0.19, 0.225, 86464) # Define time vector
#t = np.append(np.linspace(-0.2, -0.1 - (-0.1+0.2)/86400, 86400), np.linspace(-0.1, 0.2, 76640))


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
ax.plot(output.times, output.expect[2]);
ax.plot(output.times, output.expect[3]);
ax.set_xlabel('Time');
ax.set_ylabel('Expectation values');
ax.legend(("excitation no","0_occ", "Nmax_occ",));
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
ax.plot(output.times, output.expect[0]);
ax.set_xlabel('Time');
ax.set_ylabel('excitations');
ax.legend(("ee"));
show()



fig, ax = subplots()
ax.plot( Fs,output.expect[0] );
ax.set_xlabel('F');
ax.set_ylabel('n');
ax.legend(("F vs n"));
show()


#sp.io.savemat('/home/ssaumya7/Desktop/detect/qutip/works/po2_B.mat', mdict={'Fs': Fs,'t': t,'bdb': output.expect[0],'b0p': output.expect[1],'bNp': output.expect[2]})
