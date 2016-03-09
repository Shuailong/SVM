#!/usr/bin/env python
# encoding: utf-8

"""
qp.py
 
Created by Shuailong on 2016-03-08.

Qudratic Programming to solve SVM with errors and offset.

Reference: http://www.gurobi.com/documentation/6.5/examples/qp_py.html

"""

from gurobipy import *
import numpy as np

def primal(C, X, y):
    '''
    Primal form of SVM.

    minimize 
        1/2 * lambda*(theta.^2) + (xi(1)+...+xi(n))
    subject to 
        y(t) * (theta .* x(t) + theta0) >= 1 - xi(t), t = 1 ... n
        xi(t) >= 0, t = 1...n
    '''
    lamda = 1.0/C
    M = len(X[0])
    N = len(X)

    m = Model("qp")

    theta_lbs = [-GRB.INFINITY]*M
    theta_names = ['theta' + str(i+1) for i in range(M)]
    theta = [m.addVar(lb = theta_lb, name = theta_name) for theta_lb, theta_name in zip(theta_lbs, theta_names)]
    theta0 = m.addVar(lb=-GRB.INFINITY, name="theta0")

    xi_names = ['xi' + str(i+1) for i in range(N)]
    xi = [m.addVar(name = xi_name) for xi_name in xi_names]

    m.update()

    obj = lamda/2.0*np.inner(theta, theta) + 1.0/N*sum(xi)
    m.setObjective(obj, GRB.MINIMIZE)

    for i in range(N):
        m.addConstr(y[i] * np.inner(theta, X[i]) >= 1 - xi[i])

    m.optimize()

    theta = [i.X for i in theta]
    theta0 = theta0.X

    return theta, theta0

def dual(C, X, y):
    '''
    Dual form of SVM.

    maximize
        sum(alpha(t)) - 1/2 * sum( sum( alpha(i)*alpha(j)*y(i)*y(j)*(x(i).*x(j)) ) )
    subject to
        0 <= alpha(t) <= 1/lambda, sum(alpha(t)*y(t)) = 0
    '''
    lamda = 1.0/C
    M = len(X[0])
    N = len(X)
    m = Model("qp")

    alphas = [m.addVar(ub = C, name = 'alpha'+str(i+1)) for i in range(N)]

    m.update()

    subsum = 0
    for i in range(N):
        for j in range(N):
            subsum += alphas[i] * alphas[j] * y[i] * y[j] * np.inner(X[i], X[j])

    obj = sum(alphas) - 0.5*subsum

    m.setObjective(obj, GRB.MAXIMIZE)
    m.addConstr(sum([alphas[i]*y[i] for i in range(N)]) == 0)

    m.optimize()

    theta = sum([np.multiply(alphas[i].X*y[i], X[i]) for i in range(N)])
    theta0 = np.median([y[i]-sum([alphas[j].X*y[j]*np.inner(X[i],X[j]) for j in range(N)]) for i in range(N) if alphas[i].X > 0 and alphas[i].X < C])
    
    return theta, theta0

def main():
    pass

if __name__ == '__main__':
    main()
    