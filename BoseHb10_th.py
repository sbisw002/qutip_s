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
#thermal+photon
import scipy as sp
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from scipy import io

import time
import random

then = time.time() #Time before the operations start

pb_1n0 = 0.96
pb_2n1 = 0.04
#pb_b_0 = 0.1


Nmax = 120
gamma = 1
Delta = 0*gamma
kappa = 1*gamma
lamda = 1*gamma

U = 0.1*gamma
F0 = 0
F1 = 0
F2 = 50
Sgm = 2
t0 = 8

#psi0 = tensor(basis(Nmax+1, 61), basis(3,2))
psi0 = tensor(basis(Nmax+1, 61), basis(3,2))


#print(psi0)
#b = tensor(destroy(Nmax+1), qeye(1))
b = tensor(destroy(Nmax+1), qeye(3) )

F1_F0 = Qobj( np.matrix([[0,1,0],[0,0,0],[0,0,0]]) )
F0_F1 = Qobj( np.matrix([[0,0,0],[1,0,0],[0,0,0]]) )

F1_F2 = Qobj( np.matrix([[0,0,0],[0,0,0],[0,1,0]]) )
F2_F1= Qobj( np.matrix([[0,0,0],[0,0,1],[0,0,0]]) )



F1tF0 = tensor( qeye(Nmax+1), F1_F0 )
F0tF1 = tensor( qeye(Nmax+1), F0_F1 )

F1tF2 = tensor( qeye(Nmax+1), F1_F2 )
F2tF1 = tensor( qeye(Nmax+1), F2_F1 )



bdb = tensor(num(Nmax+1),qeye(3) )


bNp = tensor(basis(Nmax+1,Nmax)*basis(Nmax+1,Nmax).dag(), qeye(3)  )
b0p = tensor(basis(Nmax+1,0)*basis(Nmax+1,0).dag(), qeye(3) )



b_ops = []  # Build collapse operators
b_ops.append(np.sqrt(gamma) * b)
b_ops.append(np.sqrt(pb_1n0*kappa) * F1tF0)
b_ops.append(np.sqrt(pb_2n1*lamda) * F1tF2)

b_ops.append(np.sqrt((1-pb_1n0)*kappa) * F0tF1)
b_ops.append(np.sqrt((1-pb_2n1)*lamda) * F2tF1)




FA = Qobj(np.matrix([[F0,0,0],[0,F1,0],[0,0,F2]]) ) 

Fop = tensor(qeye(Nmax+1), FA )
#print(FA)


H0 = tensor( Qobj(-Delta* num(Nmax+1) + U*( num(Nmax+1) ** 2 - num(Nmax+1)  )     ),qeye(3)  )  
H1 = tensor( ( destroy(Nmax+1) + create(Nmax+1) ), FA )  # time-independent term
H2 = tensor( qeye(Nmax+1), (F1_F0+F0_F1) )

t = np.linspace(0.0,15, 150) # Define time vector

def H2_coeff(tt, args=None):
    return np.sqrt(kappa)*np.exp(-( (tt-t0) /2/Sgm  ) ** 2)/np.sqrt(np.sqrt(2*np.pi*Sgm*Sgm))


H01 = H0 + H1    

H = [[H2,H2_coeff],H01]

opts = Options(store_states = True)
#output = mesolve(H, psi0, t, b_ops, [],options = opts)
output = mesolve(H01, psi0 * psi0.dag(), t, b_ops, [],options = opts)
#output = mesolve(H, psi0, t, b_ops, [bdb, b0p, bNp, Fop])
qsave(output, 'Th_ph_p96_p04_K')

#sp.io.savemat('./L3_Sgm40_F1_0_A.mat', mdict={'Fs': Fs,'t': t,'bdb': output.expect[0],'b0p': output.expect[1],'bNp': output.expect[2]})

#fig.savefig('./Sgm40_F1_0_A')

now = time.time() #Time after it finished

print("It took: ", now-then, " seconds")




