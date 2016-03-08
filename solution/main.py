#!/usr/bin/env python
# encoding: utf-8

"""
main.py
 
Created by Shuailong on 2016-03-08.

Main entry of Machine Learining Assignment 2.

"""

from dataset import read_data
from qp import primal, dual

from time import time
from gurobipy import *

def predict_point(x, theta, theta0):
    '''
    Predict the class of a data point.

    :type x: numpy.ndarray
    :type theta: numpy.ndarray
    :type theta: numpy.float64
    :return {+1, -1}
    '''
    return 0

def predict(X, theta, theta0):
    '''
    Predict the classes of a dataset X

    :type X: numpy.ndarray[numpy.ndarray]
    :type theta: numpy.ndarray
    :type theta: numpy.float64
    :return List[{+1, -1}]
    '''
    n = len(X)
    y = [0]*n
    for i in range(n):
        y[i] = predict_point(X[i], theta, theta0)
    return y

def score(y_predict, y_true):
    '''
    Calculate accuracy of predictions.

    :type y_predict: List[int]
    :type y_true: List[int]
    '''

    if len(y_predict) != len(y_true):
        raise ValueError('Check the lengths of two arrays!')

    accuracy = 0
    for i in range(len(y_predict)):
        if y_predict[i] == y_true[i]:
            accuracy += 1

    return accuracy/float(len(y_predict))

def main():
    start_time = time()

    C = 1.0

    datasets = ['A', 'B', 'C']
    for dataset in datasets:
        print 'Dataset:', dataset

        trainX, trainY = read_data(dataset, 'train')
        testX, testY = read_data(dataset, 'test')

        # Primal form SVM
        print '[Primal form]'
        optimize_func = primal
        print 'Optimizing...'
        theta, theta0 = optimize_func(C, trainX, trainY)
        print 'Pridicting...'
        train_Y = predict(trainX, theta, theta0)
        train_score = score(train_Y, trainY)
        test_Y = predict(testX, theta, theta0)
        test_score = score(test_Y, testY)
        print 'Training/test accuracy:', str(round(train_score*100, 2)) + '%', '/', str(round(test_score*100, 2)) + '%'

        # Dual form SVM
        print '[Dual form]'
        optimize_func = primal
        print 'Optimizing...'
        theta, theta0 = optimize_func(C, trainX, trainY)
        print 'Pridicting...'
        train_Y = predict(trainX, theta, theta0)
        train_score = score(train_Y, trainY)
        test_Y = predict(testX, theta, theta0)
        test_score = score(test_Y, testY)
        print 'Training/test accuracy:', str(round(train_score*100, 2)) + '%', '/', str(round(test_score*100, 2)) + '%'

        print

    print '----------' + str(round(time() - start_time, 2)) + ' seconds.----------'


if __name__ == '__main__':
    main()
    