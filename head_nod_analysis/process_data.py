#
# データ処理スレッド
#

import numpy as np
import collections
import pandas as pd
import socket
import pickle

# 自作ライブラリ
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from . import add_data, get_feature, setup_variable, stop
from paz.backend import camera as CML


analysis_csv = [setup_variable.analysis_columns]  # windowデータの追加
answer_list = []  # 正解データリスト（windowごと）
feature_list = []  # 特徴量抽出データ
window_num = 0  # window番号
realtime_pred = []

# ================================= パスの取得 ================================ #
path = setup_variable.path
server_address = '192.168.2.111'


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

    return window_T.T, answer_list[-1]


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
push_server = []
check_num = 0


def Realtime_analysis(to_server=False, get_face=False):
    global window_num, feature_list, push_server, check_num
    filename = path + '\\data_set\\analysis_files\\feature_files\\feature_list1.csv'
    train_x = pd.read_csv(filename, header=None)
    filename = path + '\\data_set\\analysis_files\\answer_files\\answer_list1.csv'
    y = np.loadtxt(filename, delimiter=",", dtype='int')
    train_y = pd.Series(data=y)

    # 主成分分析 (PCA)
    pca = PCA(0.99)
    pca.fit(train_x)
    train_x = pca.transform(train_x)

    # ランダムフォレスト
    clf = RandomForestClassifier(max_depth=30, n_estimators=30, random_state=42)
    clf.fit(train_x, train_y)

    if to_server:
        host = server_address  # サーバーのホスト名
        port = 50000  # 49152~65535

        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # オブジェクトの作成をします
        client.connect((host, port))  # これでサーバーに接続します

    while stop.stop_flg:
        window = []
        window = add_data.sensor.process_window()
        if window:
            window_num += 1

            # ウィンドウラベルの付与，正解ラベルデータの作成
            result_window, answer_num = label_shape(window)
            analysis_csv.extend(result_window)

            # リアルタイム行動分析
            feature_list.append(get_feature.get_feature(window))
            test_x = pd.DataFrame(feature_list)

            test_x = pca.transform(test_x)

            y_pred = clf.predict(test_x)
            print(y_pred, answer_num)  # 判定された行動の出力
            realtime_pred.extend(y_pred)
            feature_list = []

            # 判定された表情の出力
            if get_face:
                pred_face = CML.process_window()
                print(pred_face)

            # サーバーへの送信
            if to_server:
                push_server = [y_pred[0], pred_face]
                massage = pickle.dumps(push_server)
                client.send(massage)  # 適当なデータを送信します（届く側にわかるように）
            check_num += 1
    print(check_num)
    print(realtime_pred)
    print(answer_list)
    test_y = pd.Series(data=answer_list)
    y_pred = pd.Series(data=realtime_pred)
    print(classification_report(test_y, y_pred, target_names=['others', 'nod', 'shake']))
