#!/usr/bin/env python
# encoding: utf-8

"""
main.py
 
Created by Shuailong on 2016-03-08.

Main entry of Machine Learining Assignment 2.

"""

from dataset import read_data
import mysvm

from sklearn import svm
from time import time
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt

def score(y_predict, y_true):
    '''
    Calculate accuracy of predictions.

    :type y_predict: List[int]
    :type y_true: List[int]
    '''

    if len(y_predict) != len(y_true):
        raise ValueError('Check the lengths of two arrays: ' + str(len(y_predict)) + ' != ' + str(len(y_true)))

    accuracy = 0
    for i in range(len(y_predict)):
        if y_predict[i] == y_true[i]:
            accuracy += 1

    return accuracy/float(len(y_predict))

def find_optimal_C(trainX, trainY):
    N = len(trainX)
    M = len(trainX[0])

    trainXPos = trainX[trainY == 1]
    trainXNeg = trainX[trainY == -1]

    if len(trainXPos) == 0 or len(trainXNeg) == 0:
        raise ValueError('Only one class in training set!')
        pass

    ordersPos = range(len(trainXPos))
    shuffle(ordersPos)
    ordersNeg = range(len(trainXNeg))
    shuffle(ordersNeg)

    cvX = np.concatenate([trainXPos[ordersPos[:len(trainXPos)/5]], trainXNeg[ordersNeg[:len(trainXNeg)/5]]])
    cvY = np.asarray([1] * (len(trainXPos)/5) + [-1] * (len(trainXNeg)/5))
    
    trainX = np.concatenate([trainXPos[ordersPos[len(trainXPos)/5:len(trainXPos)]], trainXNeg[ordersNeg[len(trainXNeg)/5:len(trainXNeg)]]])
    trainY = np.asarray([1] * (len(trainXPos)*4/5) + [-1] * (len(trainXNeg)*4/5))
    
    optimal_C = []
    optimal_cv_accuracy = 0

    C = 0.001
    MAX_C = 1000
    while C < MAX_C:

        clf = mysvm.SVC(C=C)
        clf.fit(trainX, trainY)

        train_Y = clf.predict(trainX)
        train_score = score(train_Y, trainY)
        cv_Y = clf.predict(cvX)
        cv_score = score(cv_Y, cvY)

        if cv_score > optimal_cv_accuracy:
            optimal_cv_accuracy = cv_score
            optimal_C = [C]
        elif cv_score == optimal_cv_accuracy:
            optimal_C.append(C)

        C *= 2

    optimal_C = np.median(np.asarray(optimal_C))
    print 'Best C/CV accuracy:', optimal_C, '/', str(round(optimal_cv_accuracy*100, 2))+'%'

    return optimal_C


def main():
    start_time = time()

    for dataset in ['A', 'B', 'C']:

        print 'Dataset:', dataset

        trainX, trainY = read_data(dataset, 'train')
        testX, testY = read_data(dataset, 'test')

        # trainXPos = trainX[trainY == 1]
        # trainXNeg = trainX[trainY == -1]

        # plt.plot(trainXPos[:,0], trainXPos[:,1], 'ro')
        # plt.plot(trainXNeg[:,0], trainXNeg[:,1], 'bo')

        # plt.show()
        # continue

        optimal_C = find_optimal_C(trainX, trainY)
        # C = 0.001
        # while C <= 1000:    
        #     print 'C:', C
        clf = mysvm.SVC(C=optimal_C, is_dual=False)
        # clf = svm.SVC(C=optimal_C)
        clf.fit(trainX, trainY)
        train_Y = clf.predict(trainX)
        train_score = score(train_Y, trainY)
        test_Y = clf.predict(testX)
        test_score = score(test_Y, testY)

        # print 'Number of SVs:', sum(clf.n_support_)
        # print 'Margin:', clf.margin

        print '---------------------------------------'
        print 'Training/test accuracy:', str(round(train_score*100, 2)) + '%', '/', str(round(test_score*100, 2)) + '%'
        print '---------------------------------------'
        print 
            # C *= 10


    print '----------' + str(round(time() - start_time, 2)) + ' seconds.---------------'


if __name__ == '__main__':
    main()
    