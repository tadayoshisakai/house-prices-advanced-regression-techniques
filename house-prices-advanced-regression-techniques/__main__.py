# numpy , pandas
import numpy as np 
import pandas as pd
# scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# 可視化用ライブラリ
import matplotlib.pyplot as plt
import seaborn as sns

def main():

    rawTrainDataFrame = pd.read_csv('../data/train.csv')
    rawTestDataFrame = pd.read_csv('../data/test.csv')
    
    distPlot(rawTrainDataFrame['SalePrice'], "Train")
    trainDataConversion = convertDataFrame(rawTrainDataFrame)
    trainDataConversion.num2str()
    trainDataConversion.completeDefectValue()
    trainDataConversion.onehotEncoding()
    trainDataConversion.outlierException()
    convTrainDataFrame = trainDataConversion.getResult()
    
    distPlot(convTrainDataFrame['SalePrice'],"Train_mod")
    print(convTrainDataFrame)
    
    print("end")

def distPlot(pltDataSeries,dataSetName = "" ):
    sns.distplot(pltDataSeries)
    plt.show()
    print("==========================\n(DistInfo)" + dataSetName + " / " +pltDataSeries.name)
    print(pltDataSeries.describe())
    print(f"skewness: {round(pltDataSeries.skew(),4)}" )
    print(f"kurtosis: {round(pltDataSeries.kurt(),4)}" )
    print("==========================")
    
class convertDataFrame:
    def __init__(self,df):
        self.df = df
        self.config = {'num2str_list':['MSSubClass','YrSold','MoSold']\
                       ,'outlierException_list':['LotArea<20000'\
                                                 ,'SalePrice<400000'\
                                                 ,'YearBuilt>1920']}
    def getResult(self):
        self._exportResultAsCSV()
        return self.df
    def num2str(self):
        for column in self.config['num2str_list']:
            self.df[column] = self.df[column].astype(str)
    def completeDefectValue(self):
        for column in self.df.columns:
            if self.df[column].dtype=='O': #0:Object
                self.df[column] = self.df[column].fillna('None')
            else:
                self.df[column] = self.df[column].fillna(0)
    def onehotEncoding(self):
        self.df = pd.get_dummies(self.df)
    def outlierException(self):
        for e in self.config['outlierException_list']:
            self.df = self.df.query(e)
    def _exportResultAsCSV(self):
        self.df.to_csv("convertedResult.csv")
if __name__=='__main__':
    main()
