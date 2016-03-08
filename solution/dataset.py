#!/usr/bin/env python
# encoding: utf-8

"""
dataset.py
 
Created by Shuailong on 2016-03-08.

Dataset for Assignment 2.

"""

import os
import numpy as np 

BASE_PATH = '../data'

def read_data(dataset_name, test_or_train):
    """
    :dataset_name: A | B | C, str
    :test_or_train: test | train, str
    :return: corresponding data
    :rtype: numpy.array
    """
    dataset_names = ['A', 'B', 'C']
    test_or_trains = ['test', 'train']
    if dataset_name not in dataset_names or test_or_train not in test_or_trains:
        raise ValueError("Check your arguments: " + dataset_name + ', ' + test_or_train)
    filename = os.path.join(BASE_PATH, dataset_name, test_or_train + '.csv')
    data = np.loadtxt(open(filename,"rb"),delimiter=" ",skiprows=0)
    dataX = data[:,:2]
    dataY = data[:,2]
    return (dataX, dataY)

def main():
    dataX, dataY = read_data('A', 'train')
    print len(dataX), len(dataY)
    print dataX[:10], dataY[:10]

if __name__ == '__main__':
    main()
    
