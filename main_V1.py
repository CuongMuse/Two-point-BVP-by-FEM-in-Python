#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Thu Mar 26 09:00:33 2020

@author: Cuong T. Nguyen (cuong.nt@vgu.edu.vn)
         Lecturer in Mechanics 
         Mechanical and Computational Engineering
         Vietnamese-German University 
         
A simple FEM code to solve a two-point b.v.p. discussed in 
the book "Finite Elements An Introduction" by Becker et. al.

Here is the strong form: 
    
    Finding u = u(x), x \in [0,1] satisfies
        -u" + u = x, x \in (0,1) 
        & B.C.'s: u(0) = u(1) = 0
    
The exact solution is given by u(x) = x - (sinh x/sinh 1)
'''

import os
os.system('clear')

print(__doc__)

import numpy as np
import math
import pylab

N = 10
h = 1/N
n = N + 1
K = np.zeros((n,n))
f = np.zeros((n,1))

# Topology matrix
edof = np.zeros((N,2),dtype=np.int)
for i in range(0,N):
    edof[i,:] = np.array([i+1,i+2])

def elm_stiff(h):
    return np.array([[1/h+h/3,-1/h+h/6],[-1/h+h/6,1/h+h/3]])

def elm_load(h,xi,xj):
    return h/6*np.array([[2*xi + xj],[xi + 2*xj]])

def assem(i,K,Ke,f,fe):
    i0 = edof[i,0]-1
    i1 = edof[i,1]
    K[i0:i1,i0:i1] = K[i0:i1,i0:i1] + Ke;
    f[i0:i1]       = f[i0:i1] + fe;
    return K, f

def solveq(K,f,bc):
    # boundary conditions
    udof = bc[:,0]
    uval = bc[:,1]
    # partition to Krr and Kuu
    n    = K.shape[0]
    rdof = np.arange(n) 
    rdof = np.delete(rdof,udof-1,axis=0)
    Krr  = K[rdof[0]:rdof[-1]+1,rdof[0]:rdof[-1]+1]
    fr   = f[rdof]
    # solve the linear system of eqns
    ur = np.linalg.solve(Krr,fr)
    # assign the soln to ur and uval
    u       = np.zeros((n,1))
    u[rdof] = ur;
    u[np.ix_(udof-1)] = np.asmatrix(uval).reshape(uval.shape[0],1)
    return u

Ke = elm_stiff(h)
for i in range(0,N):
    xi = i*h
    xj = xi + h
    fe = elm_load(h,xi,xj)
    K,f = assem(i,K,Ke,f,fe)

bc = np.array([[1,0],[n,0]])
u  = solveq(K,f,bc)
print(' The solution u = ','\n',u,'\n')

x = np.arange(0,1+h,h)
uexact = x - np.sinh(x)/math.sinh(1)

pylab.plot(x, u,'o',label='FEM') 
pylab.plot(x, uexact,label='exact') 
pylab.ylabel('y')
pylab.xlabel('x') 
pylab.show()