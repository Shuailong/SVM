#!/usr/bin/env python
# encoding: utf-8

"""
logistic_regression.py
 
Created by Shuailong on 2016-03-08.

Logistic Regression Classifier.

References:
    1. http://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/
    2. http://research.microsoft.com/en-us/um/people/minka/papers/logreg/minka-logreg.pdf
    3. http://lingpipe-blog.com/2012/02/16/howprevent-overflow-underflow-logistic-regression/

Issues:
    1. When lambda is 0.1 or larger, overflow or underflow problems may occur.
    
"""

from dataset import read_data
from main import score

import numpy as np
from random import shuffle
from math import exp, log

class LogisticRegression:

    def __init__(self, max_iter=1000, learning_rate=1.0, lamda=1.0, tol=1e-3):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.lamda = lamda
        self.tol = tol

        self.max_real_num = 50

    def loss(self, X, y):
        res = 0
        for i in range(len(X)):
            real_num = -y[i]*(np.inner(self.theta, X[i]) + self.theta0)
            if real_num > self.max_real_num:
                real_num = self.max_real_num
            elif real_num < -self.max_real_num:
                real_num = -self.max_real_num
            res += log(1 + exp(real_num))
        res += self.lamda / 2.0 * np.inner(self.theta, self.theta)

        return res

    def fit(self, X, y):
        '''
        minimize
            sum(log(1 + exp(-y[i] * (theta.* X[i] + theta0)))) + 2/lamda*(theta.*theta)
        subject to
            theta, theta0: [-inf, inf]
        '''
        M = len(X[0])
        N = len(X)

        self.theta = np.zeros(M)
        self.theta0 = 0

        orders = range(N)

        last_loss = 0
        loss = 0
        early_stop = False
        for i in range(self.max_iter): 
            shuffle(orders)
            for j in orders:
                real_num = y[j]*(np.inner(self.theta, X[j]) + self.theta0)
                if real_num > self.max_real_num:
                    real_num = self.max_real_num
                elif real_num < -self.max_real_num:
                    real_num = -self.max_real_num

                self.theta -= self.learning_rate * (-np.multiply(y[j]/(1+exp(real_num)), X[j]) + np.multiply(self.lamda, self.theta))
                self.theta0 -= self.learning_rate * (-y[j]/(1+exp(real_num)))

            loss = self.loss(X, y)
            if abs(loss-last_loss) < self.tol:
                print 'Iterations/Loss', i, '/', loss
                early_stop = True
                break
            last_loss = loss
        if not early_stop:
            print 'Iterations/Loss', self.max_iter, '/', loss


    def predict(self, X):
        return [1 if np.inner(self.theta, x) + self.theta0 > 0 else -1 for x in X]

def main():
    trainX, trainY = read_data('A', 'train')
    testX, testY = read_data('A', 'test')

    lr = LogisticRegression(max_iter=100)
    lr.fit(trainX, trainY)
    train_Y = lr.predict(trainX)
    train_score = score(train_Y, trainY)
    test_Y = lr.predict(testX)
    test_score = score(test_Y, testY)

    print '---------------------------------------'
    print 'Training/test accuracy:', str(round(train_score*100, 2)) + '%', '/', str(round(test_score*100, 2)) + '%'
    print '---------------------------------------'

if __name__ == '__main__':
    main()
    
