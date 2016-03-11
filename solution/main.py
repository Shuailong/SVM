#!/usr/bin/env python
# encoding: utf-8

"""
main.py
 
Created by Shuailong on 2016-03-08.

Main entry of Machine Learining Assignment 2.

"""

from dataset import read_data
import mysvm
from vis import plot

from time import time
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

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

    for dataset in ['A','B','C']:

        print 'Dataset:', dataset

        trainX, trainY = read_data(dataset, 'train')
        testX, testY = read_data(dataset, 'test')

        N = len(trainX)
        M = len(trainX[0])

        orders = range(N)
        shuffle(orders)
        cvX = np.asarray([trainX[orders[i], :] for i in range(N/5)])
        cvY = np.asarray([trainY[orders[i]] for i in range(N/5)])

        trainX = np.asarray([trainX[orders[i], :] for i in range(N/5, N)])
        trainY = np.asarray([trainY[orders[i]] for i in range(N/5, N)])

        trainXPos = np.asarray([trainX[i, :] for i in range(len(trainX)) if trainY[i] == 1])
        trainXNeg = np.asarray([trainX[i, :] for i in range(len(trainX)) if trainY[i] == -1])

        if len(trainXPos) == 0 or len(trainXNeg) == 0:
            # raise ValueError('Only one class in training set!')
            pass

        print 'Training size/Cross validation size:', len(trainX), '/', len(cvX)

        best_C = []
        best_cv_accuracy = 0

        C = 0.001
        MAX_C = 10000000000
        while C < MAX_C:
            # print 'C =', C

            # print 'Optimizing...'

            clf = mysvm.SVC(C=C)
            clf.fit(trainX, trainY)

            # print 'Predicting...'
            train_Y = clf.predict(trainX)
            train_score = score(train_Y, trainY)
            cv_Y = clf.predict(cvX)
            cv_score = score(cv_Y, cvY)

            if cv_score > best_cv_accuracy:
                best_cv_accuracy = cv_score
                best_C = [C]
            elif cv_score == best_cv_accuracy:
                best_C.append(C)

            C *= 10

        best_C = np.median(np.asarray(best_C))
        print 'Best C/CV accuracy:', best_C, '/', best_cv_accuracy

        clf = mysvm.SVC(C=best_C)
        clf.fit(trainX, trainY)

        train_Y = clf.predict(trainX)
        train_score = score(train_Y, trainY)
        test_Y = clf.predict(testX)
        test_score = score(test_Y, testY)

        print '---------------------------------------'
        print 'Training/test accuracy:', str(round(train_score*100, 2)) + '%', '/', str(round(test_score*100, 2)) + '%'
        print '---------------------------------------'
        print 


    print '----------' + str(round(time() - start_time, 2)) + ' seconds.---------------'


if __name__ == '__main__':
    main()
    