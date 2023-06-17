
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

def lasso_tuning(train_x,train_y):
    # alphaパラメータのリスト
    param_list = [0.001, 0.01, 0.1, 1.0, 10.0,100.0,1000.0] 

    for cnt,alpha in enumerate(param_list):
        # パラメータを設定したラッソ回帰モデル
        lasso = Lasso(alpha=alpha) 
        # パイプライン生成
        pipeline = make_pipeline(StandardScaler(), lasso)

        # 学習データ内でホールドアウト検証のために分割 テストデータの割合は0.3 seed値を0に固定
        X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.3, random_state=0)

        # 学習
        pipeline.fit(X_train,y_train)

        # RMSE(平均誤差)を計算
        train_rmse = np.sqrt(mean_squared_error(y_train, pipeline.predict(X_train)))
        test_rmse = np.sqrt(mean_squared_error(y_test, pipeline.predict(X_test)))
        # ベストパラメータを更新
        if cnt == 0:
            best_score = test_rmse
            best_param = alpha
        elif best_score > test_rmse:
            best_score = test_rmse
            best_param = alpha

    # ベストパラメータのalphaと、そのときのMSEを出力
    print('alpha : ' + str(best_param))
    print('test score is : ' +str(round(best_score,4)))

    # ベストパラメータを返却
    return best_param




#pandasのカラムが100列まで見れるようにする
pd.set_option('display.max_columns', 100)

#学習データ読み込み
# 学習データの読み込み
trainDF_raw = pd.read_csv('../data/train.csv',index_col=0)
#テストデータの読み込み
testDF_raw = pd.read_csv('../data/test.csv',index_col=0)
# 先頭5行をみてみる。
print(trainDF_raw.head())

# 売却価格のヒストグラム
#sns.distplot(trainDF_raw['SalePrice'])
# 売却価格の概要をみてみる
print(trainDF_raw["SalePrice"].describe())
print(f"歪度: {round(trainDF_raw['SalePrice'].skew(),4)}" )
print(f"尖度: {round(trainDF_raw['SalePrice'].kurt(),4)}" )

# 学習データの説明変数と、予測用データを結合
all_df = pd.concat([trainDF_raw.drop(columns='SalePrice'),testDF_raw])

num2str_list = ['MSSubClass','YrSold','MoSold']
for column in num2str_list:
    all_df[column] = all_df[column].astype(str)
    
#欠損値の補完==================================
# 変数の型ごとに欠損値の扱いが異なるため、変数ごとに処理
for column in all_df.columns:
    # dtypeがobjectの場合、文字列の変数
    if all_df[column].dtype=='O':
        all_df[column] = all_df[column].fillna('None')
    # dtypeがint , floatの場合、数字の変数
    else:
        all_df[column] = all_df[column].fillna(0)
        
#特徴量エンジニアリング==========================
# 特徴量エンジニアリングによりカラムを追加する関数
def add_new_columns(df):
    # 建物内の総面積 = 1階の面積 + 2階の面積 + 地下の面積
    df["TotalSF"] = df["1stFlrSF"] + df["2ndFlrSF"] + df["TotalBsmtSF"]

    # 一部屋あたりの平均面積 = 建物の総面積 / 部屋数
    df['AreaPerRoom'] = df['TotalSF']/df['TotRmsAbvGrd']

    # 築年数 + 最新リフォーム年 : この値が大きいほど値段が高くなりそう
    df['YearBuiltPlusRemod']=df['YearBuilt']+df['YearRemodAdd']

    # お風呂の総面積
    # Full bath : 浴槽、シャワー、洗面台、便器全てが備わったバスルーム
    # Half bath : 洗面台、便器が備わった部屋)(シャワールームがある場合もある)
    # シャワーがない場合を想定してHalf Bathには0.5の係数をつける
    df['TotalBathrooms'] = (df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath']))

    # 合計の屋根付きの玄関の総面積 
    # Porch : 屋根付きの玄関 日本風にいうと縁側
    df['TotalPorchSF'] = (df['OpenPorchSF'] + df['3SsnPorch'] + df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF'])

    # プールの有無
    df['HasPool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

    # 2階の有無
    df['Has2ndFloor'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

    # ガレージの有無
    df['HasGarage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

    # 地下室の有無
    df['HasBsmt'] = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

    # 暖炉の有無
    df['HasFireplace'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

# カラムを追加
add_new_columns(all_df)

#文字列をカテゴリカル変数化(One-hot encoding)=====================
# pd.get_dummiesを使うとカテゴリ変数化できる。
all_df = pd.get_dummies(all_df)
all_df.head()

#外れ値の除去==================================================
# pd.get_dummiesを使うとカテゴリ変数化できる。
# 学習データと予測データに分割して元のデータフレームに戻す。
trainDF_mod = pd.merge(all_df.iloc[trainDF_raw.index[0]:trainDF_raw.index[-1]],trainDF_raw['SalePrice'],left_index=True,right_index=True)
testDF_mod = all_df.iloc[trainDF_raw.index[-1]:]

#以下の条件に当てはまるものを外れ値として定義し、該当する物件のデータを削除。
#物件の価格が400,000ドル以上
#敷地面積が20,000平方メートル以上
#建築年が1920年より前
trainDF_mod = trainDF_mod[(trainDF_raw['LotArea']<20000) & (trainDF_raw['SalePrice']<400000)& (trainDF_raw['YearBuilt']>1920)]

#住宅価格の対数変換============================================
# 対数変換前のヒストグラム、歪度、尖度
sns.distplot(trainDF_raw['SalePrice'])
plt.show()
print(f"歪度: {round(trainDF_raw['SalePrice'].skew(),4)}" )
print(f"尖度: {round(trainDF_raw['SalePrice'].kurt(),4)}" )
# SalePriceLogに対数変換した値を入れる。説明の都合上新たなカラムを作るが、基本的にそのまま代入して良い。
# np.log()は底がeの対数変換を行う。
trainDF_mod['SalePriceLog'] = np.log(trainDF_raw['SalePrice'])
# 対数変換後のヒストグラム、歪度、尖度
sns.distplot(trainDF_mod['SalePriceLog'])
plt.show()
print(f"歪度: {round(trainDF_mod['SalePriceLog'].skew(),4)}" )
print(f"尖度: {round(trainDF_mod['SalePriceLog'].kurt(),4)}" )

#学習データの説明変数と目的変数、予測データの説明変数にデータフレームを分割する。
# 学習データ、説明変数
train_X = trainDF_mod.drop(columns = ['SalePrice','SalePriceLog'])
# 学習データ、目的変数
train_y = trainDF_mod['SalePriceLog']

# 予測データ、目的変数
test_X = testDF_mod


#================================
# best_alphaにベストパラメータのalphaが渡される。
best_alpha = lasso_tuning(train_X,train_y)

#================================
# ラッソ回帰モデルにベストパラメータを設定
lasso = Lasso(alpha = best_alpha)
# パイプラインの作成
pipeline = make_pipeline(StandardScaler(), lasso)
# 学習
pipeline.fit(train_X,train_y)

#===============================
# 結果を予測
pred = pipeline.predict(test_X)

#===============================
# 予測結果のプロット
sns.distplot(pred)
plt.show()
# 歪度と尖度
print(f"歪度: {round(pd.Series(pred).skew(),4)}" )
print(f"尖度: {round(pd.Series(pred).kurt(),4)}" )

#===============================
# 指数変換
pred_exp = np.exp(pred)
# 指数変換した予測結果をプロット
sns.distplot(pred_exp)
plt.show()
# 歪度と尖度
print(f"歪度: {round(pd.Series(pred_exp).skew(),4)}" )
print(f"尖度: {round(pd.Series(pred_exp).kurt(),4)}" )

#===============================
# 400,000より高い物件は除去
pred_exp_ex_outliars = pred_exp[pred_exp<400000]
# 指数変換した予測結果をプロット
sns.distplot(pred_exp_ex_outliars)
plt.show()
# 歪度と尖度
print(f"歪度: {round(pd.Series(pred_exp_ex_outliars).skew(),4)}" )
print(f"尖度: {round(pd.Series(pred_exp_ex_outliars).kurt(),4)}" )


#===============================
# 学習データの住宅価格をプロット(外れ値除去済み)
sns.distplot(trainDF_mod['SalePrice'])
plt.show()
# 歪度と尖度
print(f"歪度: {round(pd.Series(trainDF_mod['SalePrice']).skew(),4)}" )
print(f"尖度: {round(pd.Series(trainDF_mod['SalePrice']).kurt(),4)}" )

#===============================
# sample_submission.csvの読み込み
submission_df = pd.read_csv('../data/sample_submission.csv')
# sample_submission.csvの形式を確認するために先頭五行を見てみる。
submission_df.head()

#===============================
# 指数変換した値を代入
submission_df['SalePrice'] = pred_exp

# submission.csvを出力
submission_df.to_csv('submission.csv',index=False)
