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

Nmax = 120
gamma = 1
Delta = 0*gamma
kappa = 1*gamma
lamda = 1*gamma

U = 0.1*gamma
F0 = 0
F1 = 0
F2 = 50
Sgm = 20
t0 = 50
psi0 = tensor(basis(Nmax+1, 0), basis(3,0))
#print(psi0)
#b = tensor(destroy(Nmax+1), qeye(1))
b = tensor(destroy(Nmax+1), qeye(3) )

F1_F0 = Qobj( np.matrix([[0,1,0],[0,0,0],[0,0,0]]) )
F1_F2 = Qobj( np.matrix([[0,0,0],[0,0,0],[0,1,0]]) )
F0_F1 = Qobj( np.matrix([[0,0,0],[1,0,0],[0,0,0]]) )

F1tF0 = tensor( qeye(Nmax+1), F1_F0 )
F0tF1 = tensor( qeye(Nmax+1), F0_F1 )

F1tF2 = tensor( qeye(Nmax+1), F1_F2 )



bdb = tensor(num(Nmax+1),qeye(3) )


bNp = tensor(basis(Nmax+1,Nmax)*basis(Nmax+1,Nmax).dag(), qeye(3)  )
b0p = tensor(basis(Nmax+1,0)*basis(Nmax+1,0).dag(), qeye(3) )



b_ops = []  # Build collapse operators
b_ops.append(np.sqrt(gamma) * b)
b_ops.append(np.sqrt(kappa) * F1tF0)
b_ops.append(np.sqrt(lamda) * F1tF2)
FA = Qobj(np.matrix([[F0,0,0],[0,F1,0],[0,0,F2]]) ) 

Fop = tensor(qeye(Nmax+1), FA )
#print(FA)


H0 = tensor( Qobj(-Delta* num(Nmax+1) + U*( num(Nmax+1) ** 2 - num(Nmax+1)  )     ),qeye(3)  )  
H1 = tensor( ( destroy(Nmax+1) + create(Nmax+1) ), FA )  # time-independent term
H2 = tensor( qeye(Nmax+1), (F1_F0+F0_F1) )
#H2 = tensor( qeye(Nmax+1),create(2) )

t = np.linspace(0.0,100, 8000) # Define time vector
#t = np.append(np.linspace(-0.2, -0.1 - (-0.1+0.2)/86400, 86400), np.linspace(-0.1, 0.2, 7645640))


#print(H0)
#print(H1)
#print(H2)

def H2_coeff(tt, args=None):
    return np.sqrt(kappa)*np.exp(-( (tt-t0) /2/Sgm  ) ** 2)/np.sqrt(np.sqrt(2*np.pi*Sgm*Sgm))



H01 = H0 + H1    
#H = [H01,[H2,H2_coeff]]
#H = [[H2, wc_t], [H2, w1_t], H01]
#H = [[H2, w1_t],[H2, w2_t], H01]

#H = [[H2, w1_t],H01]
H = [[H2,H2_coeff],H01]
#H = [[H2, w1_t],H01]
Fs = H2_coeff(t,0)

fig, ax = subplots()
ax.plot(t, Fs);
ax.set_xlabel('Time');
ax.set_ylabel('F');
ax.legend(("F"));
show()


#rho0 = steadystate(H01, b_ops)
#output = mesolve(H, psi0, t, b_ops, [bdb, b0p, bNp, Fop])
output = mesolve(H, psi0, t, b_ops, [bdb, b0p, bNp, Fop])





#sp.io.savemat('./L3_Sgm20_F1_0.mat', mdict={'Fs': Fs,'t': t,'bdb': output.expect[0],'b0p': output.expect[1],'bNp': output.expect[2]})

#fig, ax = subplots()
#ax.plot(output.times, output.expect[0]);
#ax.plot(output.times, output.expect[3]);
#ax.set_xlabel('Time');
#ax.set_ylabel('for Sigma(of phi(t))=20');
#ax.legend(("excitations","<F>"));
#show()
#fig.savefig('./Sgm20_F1_0')



