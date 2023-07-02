#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 10:53:32 2023

@author: sakaitadayoshi
"""

# numpy , pandas
import numpy as np 
import pandas as pd
from convert_data_JSON import convert_data_json

def main():

    df = pd.read_csv('../data/train.csv')
    conv = convert_data(df)
    train = conv.getConvertedTrainDF()
    print("main")

class convert_data:
    def __init__(self,df):
        self.df = df
        self.config = convert_data_json()
    def _num2str(self):
        for column in self.config.num2str_list:
            self.df[column] = self.df[column].astype(str)
    def _completeDefectValueByZero(self):
        for column in self.df.columns:
            if self.df[column].dtype=='O': #0:Object
                self.df[column] = self.df[column].fillna('None')
            else:
                self.df[column] = self.df[column].fillna(0)
    def _onehotEncoding(self):
        self.df = pd.get_dummies(self.df)
    def _outlierException(self):
        for e in self.config.outlierException_list:
            self.df = self.df.query(e)
    def _test(self):
        print("test")
    def getConvertedTrainDF(self):
        for e in self.config.trainMethod_list:
            #print(e)
            eval('self.' + e)()
        return self.df
    
if __name__=="__main__":
    main()