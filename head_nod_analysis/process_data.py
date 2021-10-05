#
# データ処理スレッド
#

import numpy as np
import collections
import pandas as pd

# 自作ライブラリ
from . import add_data, get_feature, setup_variable, stop
from paz.backend import camera as CML

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

analysis_csv = [setup_variable.analysis_columns]  # windowデータの追加
answer_list = []  # 正解データリスト（windowごと）
feature_list = []  # 特徴量抽出データ
window_num = 0  # window番号
realtime_pred = []

# ================================= パスの取得 ================================ #
path = setup_variable.path


# ============================ ラベル整形スレッド ============================== #
# ウィンドウラベルの付与，正解ラベルデータの作成
def label_shape(window):
    window_T = np.array(window).T  # 転置　（labelごとの個数を計算するため）
    label_num = collections.Counter(window_T[0])  # labelごとの個数を計算

    # 正解データファイルの出力
    if label_num['nod'] > len(window) / 2:
        answer_list.append(1)
    elif label_num['shake'] > len(window) / 2:
        answer_list.append(2)
    else:
        answer_list.append(0)
    window_T[0] = [window_num] * len(window)  # window_IDの追加

    return window_T.T


# ============================ データ処理スレッド ============================== #
# 測定したデータを処理する関数
def ProcessData():
    global window_num
    while stop.stop_flg:
        window = add_data.sensor.process_window()
        if window:
            window_num += 1
            analysis_csv.extend(label_shape(window))


# 測定したデータを処理する関数
def Realtime_analysis():
    global window_num, feature_list
    filename = path + '\\data_set\\analysis_files\\feature_files\\feature_list1.csv'
    train_x = pd.read_csv(filename, header=None)
    filename = path + '\\data_set\\analysis_files\\answer_files\\answer_list1.csv'
    y = np.loadtxt(filename, delimiter=",", dtype='int')
    train_y = pd.Series(data=y)

    clf = RandomForestClassifier(max_depth=30, n_estimators=30, random_state=42)
    clf.fit(train_x, train_y)

    while stop.stop_flg:
        window = add_data.sensor.process_window()
        if window:
            window_num += 1

            # ウィンドウラベルの付与，正解ラベルデータの作成
            analysis_csv.extend(label_shape(window))

            # リアルタイム行動分析
            feature_list.append(get_feature.get_feature(window))
            X = pd.DataFrame(feature_list)
            y_pred = clf.predict(X)
            print(y_pred)  # 判定された行動の出力
            # print(CML.process_window())
            realtime_pred.extend(y_pred)
            feature_list = []

    print(realtime_pred)
    print(answer_list)
    test_y = pd.Series(data=answer_list)
    y_pred = pd.Series(data=realtime_pred)
    # print(accuracy_score(test_y, y_pred))
    print(classification_report(test_y, y_pred, target_names=['others', 'nod', 'shake']))
