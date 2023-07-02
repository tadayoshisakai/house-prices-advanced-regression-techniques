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

#自作モジュールのインポート
from load_data import load_data
from convert_data import convert_data


def main():
    trainDF, testDF = load_data().getLoadedDataFrame()
    trainDF = convert_data(trainDF).getConvertedTrainDF()
    print("End")
    
if __name__=='__main__':
    main()
