#
# データ分析スレッド
#

from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
import pandas as pd
import numpy as np


# ============================ データ分析スレッド ============================== #
# 処理したデータを分析するスレッド
if __name__ == '__main__':

    # 特徴量抽出
    df = pd.read_csv('analysis_files/analysis.csv', encoding='utf-8')
    X = extract_features(df, column_id='window_ID')
    print(X.shape)
    print(X.head(10))

    # 正解データ取得
    y = np.loadtxt('analysis_files/answer_files/answer.csv', delimiter=",", dtype='int')
    y = pd.Series(data=y)
    y.index += 1
    print(y.shape)

    # 特徴量削減
    impute(X)
    X = select_features(X, y)

    # 学習データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

    # 学習とクラス分類予測
    cl = xgb.XGBClassifier()
    cl.fit(X_train, y_train)
    print(classification_report(y_test, cl.predict(X_test)))

    # 特徴量別重要度
    importances = pd.Series(index=X_train.columns, data=cl.feature_importances_)
    print(importances.sort_values(ascending=False).head(10))






