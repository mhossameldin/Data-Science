# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 17:34:30 2014

@author: Mahmoud Aly
         mhossameldin.aly@gmail.com
         This source code could be reused or redistributed freely.
"""

import sys
import numpy as np

def readTrainingData(f, n):
    X = np.zeros([n, f], dtype=float)
    y = np.zeros([n, 1], dtype=float)
    for i in range(n):
        featureArray = raw_input().split()
        y[i] = float(featureArray[f])
        for j in range(f):
            X[i][j] = float(featureArray[j])
            
    return X, y
def readTestData(testSize , f):
    X = np.zeros([testSize, f], dtype=float)
    for i in range(testSize):
        featureArray = raw_input().split()
        for j in range(f):
            X[i][j] = float(featureArray[j])
    
    return X

def mapX(X, degree):
    size = len(X)
    mappedX = np.ones([size, 1], dtype=float)

    for i in range(1,degree+1):
        for j in range(i+1):
            newFeature = np.reshape(np.multiply( np.power(X[:,0],i-j) , np.power(X[:,1] , j) ) , (size, 1))
            mappedX = np.concatenate((mappedX, newFeature), 1)
    return mappedX
    
def normalEquations(X, y):
    return np.dot(np.linalg.pinv(np.dot(np.transpose(X), X)), np.dot(np.transpose(X),y))

def calcEstimates(Xtest, Theta):
    return np.sum(np.multiply(np.transpose(Theta), Xtest),1)

def main(*argv):
    firstRow = raw_input()
    args = firstRow.split()
    f = int(args[0])
    n = int(args[1])
    Theta = np.zeros([1 , f], dtype=float)
    
    #read the input data and put them into matrcies
    Xtrain, ytrain = readTrainingData(f, n)
    ntest = int(raw_input())
    Xtest = readTestData(ntest, f)
       
    #Map Xtrain and Xtest to higher Polynomial Functions, Cupic functions in this case
    degree = 3
    Xhigher = mapX(Xtrain, degree)
    Xthigher = mapX(Xtest, degree)
    
    #Estimate Parameters Using Normal Equations
    Theta = normalEquations(Xhigher, ytrain)
    
    #Calculate Out for testSet using the estimated parameters
    output = calcEstimates(Xthigher, Theta)
    
    #print Output
    for i in range(len(output)):
        print output[i]
    
if __name__ == '__main__':
    sys.exit(main(*sys.argv))