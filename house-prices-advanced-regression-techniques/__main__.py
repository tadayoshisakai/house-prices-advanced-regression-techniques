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

import json

def main():

    rawTrainDataFrame = pd.read_csv('../data/train.csv')
    rawTestDataFrame = pd.read_csv('../data/test.csv')
    dataDiscription = describeDataSet()

    
    #distPlot(rawTrainDataFrame['SalePrice'], "Train")
    trainDataConversion = convertDataFrame(rawTrainDataFrame)
    trainDataConversion.num2str()
    trainDataConversion.completeDefectValue()
    trainDataConversion.onehotEncoding()
    trainDataConversion.outlierException()
    
    convTrainDataFrame = trainDataConversion.getResult()
    dataDiscription.distPlot(convTrainDataFrame['SalePrice'],"TrainData")
    
    trainDataConversion.logConversionTV()
    
    convTrainDataFrame = trainDataConversion.getResult()
    dataDiscription.distPlot(convTrainDataFrame['SalePrice_Log'],"TrainData(Log)")
    #dataDiscription.plotAll(convTrainDataFrame,"TrainingData")    
    #distPlot(convTrainDataFrame['SalePrice'],"Train_mod")
    #print(convTrainDataFrame)
    
    ls = lassoRegression(convTrainDataFrame,'SalePrice',['SalePrice','SalePrice_Log'])
    ls.lasso_tuning()
    
    print("end")

#データセット可視化用クラス
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
    def scatterPlotMulti(self,df_list,x_axisName,y_axisName):
        plt.figure(figsize=(10,10))        
        for e in df_list:
            sns.scatterplot(data=e,x=x_axisName,y=y_axisName)
        plt.show()
  
#データ変換用クラス      
class convertDataFrame:
    def __init__(self,df):
        self.df = df
        with open('../config/convDataFrameConfig.json') as f:
            self.config = json.load(f)['conf_001']
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
    def logConversionTV(self):
        self.df[self.config['targetVariable']+'_Log'] = np.log(self.df[self.config['targetVariable']])
    def _exportResultAsCSV(self):
        self.df.to_csv("convertedResult.csv")
class lassoRegression:
    def __init__(self,df,targetVal,trainExcudeVal_list):
        self.df = df
        self.explainDataFrame = df.drop(columns = trainExcudeVal_list)
        self.targetDataSeries = df[targetVal]
        self.dataDescription = describeDataSet()
    def lasso_tuning(self):
        self.a_param_list = [0.001, 0.01, 0.1, 1.0, 10.0,100.0,1000.0] 
        self._splitDataSet()
        for i,a in enumerate(self.a_param_list):
            # パラメータを設定したラッソ回帰モデル
            lasso = Lasso(alpha=a)
            # パイプライン生成
            pipeline = make_pipeline(StandardScaler(), lasso)  
            # 学習
            pipeline.fit(self.X_train,self.y_train)
            pred_train = pipeline.predict(self.X_train)
            pred_test = pipeline.predict(self.X_test)
    
            # RMSE(平均誤差)を計算
            train_rmse = np.sqrt(mean_squared_error(self.y_train,pred_train))
            test_rmse = np.sqrt(mean_squared_error(self.y_test, pred_test))
            
#            plt.plot(self.X_train['Id'],self.y_train)
#            plt.plot(self.X_train['Id'],pred_train)
#            plt.show()
#
#            plt.plot(self.X_test['Id'],self.y_test)
#            plt.plot(self.X_test['Id'],pred_test)
#            plt.show()
            
            # ベストパラメータを更新
            if i == 0:
                best_score = test_rmse
                best_param = a
            elif best_score > test_rmse:
                best_score = test_rmse
                best_param = a
    
        # ベストパラメータのalphaと、そのときのMSEを出力
        print('alpha : ' + str(best_param))
        print('test score is : ' +str(round(best_score,4)))
    
        # ベストパラメータを返却
        return best_param
    def _splitDataSet(self):
            # 学習データ内でホールドアウト検証のために分割 テストデータの割合は0.3 seed値を0に固定
            self.X_train,self.X_test,self.y_train,self.y_test \
            = train_test_split(self.explainDataFrame\
                               ,self.targetDataSeries\
                               ,test_size=0.3\
                               ,random_state=0)
            self.dataDescription.distPlot(self.y_train,"Training")
            self.dataDescription.distPlot(self.y_test,"Test")

if __name__=='__main__':
    main()
