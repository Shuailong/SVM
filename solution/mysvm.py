#!/usr/bin/env python
# encoding: utf-8

"""
qp.py
 
Created by Shuailong on 2016-03-08.

Qudratic Programming to solve SVM with errors and offset.

Reference: 
    1. http://www.gurobi.com/documentation/6.5/examples/qp_py.html
    2. https://github.com/cmaes/svmexample/blob/master/svm.py
    3. http://scikit-learn.org/stable/modules/svm.html#svm-kernels

Issues:
    1. dual form is much slower than primal form. (should be faster?) 

"""

from gurobipy import *
import numpy as np
from math import exp, tanh

class SVC:

    def __init__(self, C=1.0, kernel='linear', degree=3, gamma='auto', coef0=0.0, dual=False):
        #Penalty parameter C of the error term.
        self.C = C
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        kernels = {'linear': self.linear, 'poly': self.poly, 'rbf': self.rbf, 'sigmoid': self.sigmoid}
        self.kernel = kernels.get(kernel, 'linear')
        self.dual = dual

    # Kernel functions

    def linear(self, x1, x2):
        return np.inner(x1, x2)

    def poly(self, x1, x2):
        return (np.inner(x1, x2)*self.gamma + self.coef0)**self.degree

    def rbf(self, x1, x2):
        return exp(-self.gamma*np.inner(x1-x2, x1-x2))

    def sigmoid(self, x1, x2):
        return tanh(gamma*np.inner(x1, x2) + self.coef0)


    def primal(self):
        '''
        Primal form of SVM.

        minimize 
            1/2 * lambda*(theta.^2) + (xi(1)+...+xi(n))
        subject to 
            y(t) * (theta .* x(t) + theta0) >= 1 - xi(t), t = 1 ... n
            xi(t) >= 0, t = 1...n
        '''
        lamda = 1.0/self.C
        M = len(self.X[0])
        N = len(self.X)

        m = Model("qp")

        m.setParam('OutputFlag', False) # quiet

        theta = [m.addVar(lb = -GRB.INFINITY, name = 'theta' + str(i+1)) for i in range(M)]
        theta0 = m.addVar(lb = -GRB.INFINITY, name = "theta0")

        xi = [m.addVar(name = 'xi' + str(i+1)) for i in range(N)]

        m.update()

        obj = np.inner(theta, theta)*lamda/2.0 + sum(xi)
        m.setObjective(obj, GRB.MINIMIZE)

        for i in range(N):
            m.addConstr(self.y[i] * (np.inner(theta, self.X[i]) + theta0) >= 1 - xi[i])

        m.optimize()

        theta = [i.X for i in theta]
        theta0 = theta0.X

        self.margin = 1.0/np.linalg.norm(theta) if np.linalg.norm(theta) != 0 else float('inf')

        self.coef_ = theta
        self.intercept_ = theta0

    def dual(self):
        '''
        Dual form of SVM.

        maximize
            sum(alpha(t)) - 1/2 * sum( sum( alpha(i)*alpha(j)*y(i)*y(j)*(x(i).*x(j)) ) )
        subject to
            0 <= alpha(t) <= 1/lambda, sum(alpha(t)*y(t)) = 0
        '''
        lamda = 1.0/self.C
        M = len(self.X[0])
        N = len(self.X)
        m = Model("qp")

        m.setParam('OutputFlag', False) # quiet

        alphas = [m.addVar(ub = C, name = 'alpha'+str(i+1)) for i in range(N)]

        m.update()

        subsum = 0
        for i in range(N):
            for j in range(N):
                subsum += alphas[i] * alphas[j] * self.y[i] * self.y[j] * self.kernel(X[i], X[j])

        obj = sum(alphas) - 0.5 * subsum

        m.setObjective(obj, GRB.MAXIMIZE)

        m.addConstr(sum([alphas[i]*self.y[i] for i in range(N)]) == 0)
        m.optimize()

        theta = sum([np.multiply(alphas[i].X*self.y[i], self.X[i]) for i in range(N)])

        self.support_ = np.asarray([i for i in range(N) if alphas[i].X > 1e-8 and alphas[i].X < C*0.99999])
        
        if len(self.support_) == 0:
            theta0 = 0
        else:
            theta0 = np.median([y[i]-sum([alphas[j].X*self.y[j]*self.kernel(X[i],X[j]) for j in range(N)]) for i in support_vec_idx])

        self.support_vectors_ = np.asarray([self.X[i] for i in self.support_])
        self.n_support_ = len(support_vec_idx)
        self.coef_ = theta
        self.intercept_ = theta

 
    def fit(self, X, y):
        self.X = X
        self.y = y

        if self.gamma == 'auto':
            self.gamma = 1.0/len(X[0])

        if self.dual:
            self.dual()
        else:
            self.primal()

    def predict(self, X):
        '''
        Predict the classes of a dataset X

        :type X: numpy.ndarray[numpy.ndarray]
        :return List[{+1, -1}]
        '''

        return [1 if np.inner(X[i], self.coef_) + self.intercept_ > 0 else -1 for i in range(len(X))]
 

def main():
    pass

if __name__ == '__main__':
    main()
    