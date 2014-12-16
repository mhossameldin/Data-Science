# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 20:16:41 2014

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
    
    #Add the intercept terms
    Xtrain = np.concatenate((np.ones([n, 1], dtype=float) , Xtrain),1)
    Xtest = np.concatenate((np.ones([ntest, 1], dtype=float) , Xtest),1)
    
    #Estimate Parameters Using Normal Equations
    Theta = normalEquations(Xtrain, ytrain)
    
    #Calculate Out for testSet using the estimated parameters
    output = calcEstimates(Xtest, Theta)
    
    #print Output
    for i in range(len(output)):
        print output[i]
    
if __name__ == '__main__':
    sys.exit(main(*sys.argv))

