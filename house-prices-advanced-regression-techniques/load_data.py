#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 10:35:34 2023

@author: sakaitadayoshi
"""

import pandas as pd

def main():
    trainDF, testDF = load_data().getLoadedDataFrame()
    print("main")    
    
class load_data:
    def __init__(self):
        self.train = pd.read_csv('../data/train.csv')
        self.test = pd.read_csv('../data/test.csv')
    def getLoadedDataFrame(self):
        return self.train, self.test

if __name__=="__main__":
    main()
    