#!/usr/bin/env python
# encoding: utf-8

"""
qp.py
 
Created by Shuailong on 2016-03-08.

Qudratic Programming to solve SVM with errors and offset.

Reference: 
    1. http://www.gurobi.com/documentation/6.5/examples/qp_py.html
    2. https://github.com/cmaes/svmexample/blob/master/svm.py

Issues:
    1. dual form is much slower than primal form. (should be faster?) 

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

    m.setParam('OutputFlag', False) # quiet

    theta = [m.addVar(lb = -GRB.INFINITY, name = 'theta' + str(i+1)) for i in range(M)]
    theta0 = m.addVar(lb = -GRB.INFINITY, name = "theta0")

    xi = [m.addVar(name = 'xi' + str(i+1)) for i in range(N)]

    m.update()

    obj = np.inner(theta, theta)*lamda/2.0 + sum(xi)
    m.setObjective(obj, GRB.MINIMIZE)

    for i in range(N):
        m.addConstr(y[i] * (np.inner(theta, X[i]) + theta0) >= 1 - xi[i])

    m.optimize()

    theta = [i.X for i in theta]
    theta0 = theta0.X

    margin = 1.0/np.linalg.norm(theta) if np.linalg.norm(theta) != 0 else float('inf')
    print 'Margin:', margin

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

    m.setParam('OutputFlag', False) # quiet

    alphas = [m.addVar(ub = C, name = 'alpha'+str(i+1)) for i in range(N)]

    m.update()

    subsum = 0
    for i in range(N):
        for j in range(N):
            subsum += alphas[i] * alphas[j] * y[i] * y[j] * np.inner(X[i], X[j])

    obj = sum(alphas) - 0.5 * subsum

    m.setObjective(obj, GRB.MAXIMIZE)

    m.addConstr(sum([alphas[i]*y[i] for i in range(N)]) == 0)
    m.optimize()

    theta = sum([np.multiply(alphas[i].X*y[i], X[i]) for i in range(N)])

    support_vec_idx = [i for i in range(N) if alphas[i].X > 1e-8 and alphas[i].X < C*0.99999]
    print 'Number of support vectors:', len(support_vec_idx)
    if len(support_vec_idx) == 0:
        theta0 = 0
    else:
        theta0 = np.median([y[i]-sum([alphas[j].X*y[j]*np.inner(X[i],X[j]) for j in range(N)]) for i in support_vec_idx])

    return theta, theta0

def main():
    pass

if __name__ == '__main__':
    main()
    