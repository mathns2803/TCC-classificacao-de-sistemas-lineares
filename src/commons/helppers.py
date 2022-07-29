# -*- coding: utf-8 -*-
"""
Created on Sun May 15 14:14:03 2022

@author: theus
"""
import pandas as pd
import numpy as np

ordem1 = pd.read_table(".\\dados\\fonte.txt", delimiter=r'\s+', header = None)

def readTxtFile(filePath):
    filePath = pd.read_table(filePath, delimiter=r'\s+', header = None)
    out = pd.DataFrame(data=filePath)
    out.loc[:,1] = ordem1[0]
    return out



#  This module contains functions to use like helpers for some codes

def serialize(inputArray, factor):
    outputArray = []
    for data in divideInputIn8Parts(inputArray, 50):
        print(data)
        # for d in data:
        #     outputArray.append(d)
    return np.array(outputArray)

def divideInputIn8Parts(inputArray, factor):
    outputArray = []
    outputArray.append(inputArray[0][0*factor:(len(inputArray[0])-3*factor)])
    outputArray.append(inputArray[0][1*factor:(len(inputArray[0])-2*factor)])
    outputArray.append(inputArray[0][2*factor:(len(inputArray[0])-1*factor)])
    outputArray.append(inputArray[0][3*factor:(len(inputArray[0])-0*factor)])
        
    outputArray.append(inputArray[1][0*factor:(len(inputArray[1])-3*factor)])
    outputArray.append(inputArray[1][1*factor:(len(inputArray[1])-2*factor)])
    outputArray.append(inputArray[1][2*factor:(len(inputArray[1])-1*factor)])
    outputArray.append(inputArray[1][3*factor:(len(inputArray[1])-0*factor)])
    return np.array(outputArray);


