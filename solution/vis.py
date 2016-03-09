#!/usr/bin/env python
# encoding: utf-8

"""
vis.py
 
Created by Shuailong on 2016-03-09.

Data point visualization.

"""

import matplotlib.pyplot as plt

def plot(X):
    X1 = X[:,0]
    X2 = X[:,1]
    plt.plot(X1, X2, 'ro')
    plt.show()


def main():
    pass

if __name__ == '__main__':
    main()