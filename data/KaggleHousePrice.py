#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 11:03:02 2022

@author: sakaitadayoshi
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import (
        LinearRegression,
        Ridge,
        Lasso
        )

def normalGradientDescent(X,y,theta,alpha,num_iters):
    for it in range(num_iters):
        normalCarcCostFunction
        theta=theta-alpha*np.dot(X.T,(np.dot(X,theta))-y)
    return grad    
def normalCarcCostFunction(X,y,theta):
    cost=(1/(2*m))*np.sum((np.dot(X,theta)-y)**2)
    return cost    
def gradientDescent(X,y,theta,learning_rate=0.01,iterations=5000):
    #iter_arr=range(100)
    cost_history=np.zeros(iterations)
    for it in range(iterations):
        cost, grad=calcCostFunction(X,y,theta,0.3)
        theta=theta-learning_rate*grad
        cost_history[it]=cost
    #print(cost_history)
    plt.plot(np.array(range(iterations)),cost_history)
    plt.show()
    return theta
    
def calcCostFunction(X,y,theta,mylambda):
    theta_temp=theta
    theta_temp[0]=0
    cost=(1/(2*m))*np.sum((np.dot(X,theta)-y)**2)+(mylambda/(2*m))*np.sum(np.dot(theta_temp.T,theta_temp))
    grad=(1/m)*np.dot(X.T,np.dot(X,theta)-y)+(mylambda/m)*theta_temp
    return cost,grad

#テストデータをロードする===========================
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

#学習データとテストデータをマージする==================
train['WhatIsData']='Train'
test['WhatIsData']='Test'
test['SalePrice']=9999999999
alldata=pd.concat([train,test],axis=0).reset_index(drop=True)
#print('The size of train is : ' + str(train.shape))
#print('The size of train is : ' + str(test.shape))

#欠損値を補完する=================================
#データの欠損状況を確認
#print(train.isnull().sum()[train.isnull().sum()>0].sort_values(ascending=False))
#print(test.isnull().sum()[test.isnull().sum()>0].sort_values(ascending=False))
#欠損を含むカラムのデータ型を確認
na_col_list=alldata.isnull().sum()[alldata.isnull().sum()>0].index.tolist()
#print(alldata[na_col_list].dtypes.sort_values())
#データ型に応じて欠損値を補完する
#float>>0.0
na_float_cols=alldata[na_col_list].dtypes[alldata[na_col_list].dtypes=='float64'].index.tolist()
for na_float_col in na_float_cols:
    alldata.loc[alldata[na_float_col].isnull(),na_float_col]=0.0
#object>>'NA'
na_obj_cols=alldata[na_col_list].dtypes[alldata[na_col_list].dtypes=='object'].index.tolist()
for na_obj_col in na_obj_cols:
    alldata.loc[alldata[na_obj_col].isnull(),na_obj_col]='NA'
#マージデータの欠損状況確認
#print(alldata.isnull().sum()[alldata.isnull().sum()>0].sort_values(ascending=False))

#カテゴリ変数をダミー化する==========================
#カテゴリカル変数のインデックスをリスト化する
cat_cols=alldata.dtypes[alldata.dtypes=='object'].index.tolist()
#数値変数のインデックスをリスト化する
num_cols=alldata.dtypes[alldata.dtypes!='object'].index.tolist()
#データ分割に必要なカラムをリスト化
other_cols=['Id','WhatIsData']
#説明変数でないカラムをリストから除外
cat_cols.remove('WhatIsData')
num_cols.remove('Id')
#カテゴリカル変数をダミー化する
alldata_cat=pd.get_dummies(alldata[cat_cols])
#ダミー化したデータを統合する
all_data=pd.concat([alldata[other_cols],alldata[num_cols],alldata_cat],axis=1)

#マージデータを整理する=============================
train_=all_data[all_data['WhatIsData']=='Train'].drop(['WhatIsData','Id'],axis=1).reset_index(drop=True)
test_=all_data[all_data['WhatIsData']=='Test'].drop(['WhatIsData','SalePrice'],axis=1).reset_index(drop=True)
#学習データを分割
train_x=train_.drop('SalePrice',axis=1)
train_y=train_['SalePrice']
#テストデータを分割
test_id=test_['Id']
test_data=test_.drop('Id',axis=1)
#目的変数の分布変換===============================
#sns.distplot(train['SalePrice'])
#sns.distplot(np.log(train['SalePrice']))

#スケーリング=====================================
temp_x=train_x.copy(deep=True)

for column_name in train_x:
    if train_x[column_name].max() > 1:
        #print(column_name + " : max="+str(train_x[column_name].max())+", min="+str(train_x[column_name].min()))
        train_x[column_name]=(train_x[column_name]-train_x[column_name].min())/(train_x[column_name].max()-train_x[column_name].min())

print(temp_x)
#GradientDescentで使用できる形式に変更==============
#np.ndarrayに変換
train_x=train_x.to_numpy()
#バイアス項を追加
m, n=train_x.shape
train_x=np.c_[np.ones((m,1)),train_x]
theta=np.ones((n+1,1))
train_y=train_y.to_numpy()
train_y=np.transpose(train_y)
train_y=train_y.reshape(m,1)

learnedtheta=gradientDescent(train_x,train_y,theta)
test_data=test_data.to_numpy()
m, n=test_data.shape
test_data=np.c_[np.ones((m,1)),test_data]


result=np.dot(test_data,theta)
result=pd.DataFrame(data=result,columns=['SalePrice'])
result=pd.concat([test_id,result],axis=1)

