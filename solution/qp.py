#!/usr/bin/env python
# encoding: utf-8

"""
qp.py
 
Created by Shuailong on 2016-03-08.

Qudratic Programming to solve SVM with errors and offset.

Reference: http://www.gurobi.com/documentation/6.5/examples/qp_py.html

"""

def primal(C, X, y):
    '''
    Primal form of SVM.

    minimize 
        1/2 * lambda*(theta.^2) + (xi(1)+...+xi(n))
    subject to 
        y(t) * (theta .* x(t) + theta0) >= 1 - xi(t), t = 1 ... n
        xi(t) >= 0, t = 1...n
    '''
    return 0,0

def dual(C, X, y):
    '''
    Dual form of SVM.

    maximize
        sum(alpha(t)) - 1/2 * sum( sum( alpha(i)*alpha(j)*y(i)*y(j)*(x(i).*x(j)) ) )
    subject to
        0 <= alpha(t) <= 1/lambda, sum(alpha(t)*y(t)) = 0
    '''
    return 0,0
    


def main():
    pass

if __name__ == '__main__':
    main()
    