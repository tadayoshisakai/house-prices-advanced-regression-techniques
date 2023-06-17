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
    data = describeDataSet()

    
    #distPlot(rawTrainDataFrame['SalePrice'], "Train")
    trainDataConversion = convertDataFrame(rawTrainDataFrame)
    trainDataConversion.num2str()
    trainDataConversion.completeDefectValue()
    trainDataConversion.onehotEncoding()
    trainDataConversion.outlierException()
    convTrainDataFrame = trainDataConversion.getResult()
    data.plotAll(convTrainDataFrame,"TrainingData")    
    #distPlot(convTrainDataFrame['SalePrice'],"Train_mod")
    #print(convTrainDataFrame)
    
    print("end")

class describeDataSet:
    def plotAll(self,df,dataSetName):
        for column in df.columns:
            try:
                self.distPlot(df[column],dataSetName)
                self.scatterPlot_vsSalePrice(df,df[column],dataSetName)
            except TypeError:
                print("TypeError:" + df[column].name)
            except ValueError:
                print("ValueError:" + df[column].name)
    def distPlot(self,series,dataSetName):
        sns.distplot(series)
        plt.savefig("../figure/distPlot_" + dataSetName + "_" + series.name + ".png")
        plt.show()
        print("==========================\n(DistInfo)" + dataSetName + " / " + series.name)
        print(series.describe())
        print(f"skewness: {round(series.skew(),4)}" )
        print(f"kurtosis: {round(series.kurt(),4)}" )
        print("==========================")
    def scatterPlot_vsSalePrice(self,df,series,dataSetName):
        plt.figure(figsize=(10,10))
        sns.scatterplot(data=df, x=series.name, y="SalePrice")
        plt.savefig("../figure/sccatterPlot_" + dataSetName + "_" + series.name + "_vsSalePrice.png")
        plt.show()
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
