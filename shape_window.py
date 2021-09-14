#
# データ処理スレッド
#
import csv
import glob
from _csv import reader
import numpy as np
import collections
import pandas as pd

import setup_variable
import get_feature

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
sns.set_style('whitegrid',{'linestyle.grid':'--'})


analysis_csv = [setup_variable.analysis_columns]  # windowデータの追加
answer_list = []    # 正解データリスト（windowごと）
feature_list = []   # 特徴量抽出データ


# ============================ データ整形スレッド ============================== #
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


# 生データのスケーリング
def scaling_data(df):
    # accの値をm/s^2に変換
    df['acc_X'] = df['acc_X'] / 8192 * 9.80665
    df['acc_Y'] = df['acc_Y'] / 8192 * 9.80665
    df['acc_Z'] = df['acc_Z'] / 8192 * 9.80665

    # gyroの値をdeg/sに変換
    df['gyro_X'] = df['gyro_X'] / 65.5
    df['gyro_Y'] = df['gyro_Y'] / 65.5
    df['gyro_Z'] = df['gyro_Z'] / 65.5
    return df


# 合成軸の計算
def mixed_acc(df):
    result = (df['acc_X'] ** 2 + df['acc_Y'] ** 2 + df['acc_Z'] ** 2) ** 0.5
    return result


def mixed_gyro(df):
    result = (df['gyro_X'] ** 2 + df['gyro_Y'] ** 2 + df['gyro_Z'] ** 2) ** 0.5
    return result


# ============================ ウィンドウ処理スレッド ============================== #
# ウィンドウ単位の処理用定数
T = setup_variable.T  # サンプリング周期 [Hz]
N = setup_variable.N  # ウィンドウサイズ(0.781秒)
OVERLAP = setup_variable.OVERLAP  # オーバーラップ率 [%]
window_data = []
get_window = []
window_num = 0  # window番号


# ウィンドウ処理を行う
def process_window(data_queue):
    global window_data, window_num, get_window
    not_dup = int(N * (1 - OVERLAP / 100))  # 重複しない部分の個数
    if not_dup < 1:
        not_dup = 1

    # サンプル数（N）分のデータを格納するリスト（window_data）の作成
    for _ in range(not_dup):
        # 重複しない部分のデータはキューから削除
        window_data.append(data_queue.pop(0))
    for i in range(N - not_dup):
        window_data.append(data_queue[i])

    if window_data:
        get_window = window_data
        window_data = []  # ウィンドウをリセット
        window_num += 1
        return get_window


# 各ファイルごとにウィンドウ処理を実行，結果をCSV出力
def do_process_window():
    # logファイルのコピー
    log_list = glob.glob('log_files/value_list*.csv')
    for file_name in log_list:
        with open(file_name, 'r') as f:
            csv_reader = reader(f)
            log_data = list(csv_reader)
        with open('analysis_files/data_files/' + file_name.split('\\')[1], 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerows(log_data)

    # CSVファイル出力
    file_list = glob.glob('analysis_files/data_files/value_list*.csv')
    for file_name in file_list:
        df_data_queue = pd.read_csv(file_name, names=(
            setup_variable.analysis_columns))

        # # スケーリング
        # df_data_queue = scaling_data(df_data_queue)
        # # 合成軸の計算・追加
        # df_data_queue['acc_xyz'] = df_data_queue.apply(mixed_acc, axis=1)
        # df_data_queue['gyro_xyz'] = df_data_queue.apply(mixed_gyro, axis=1)

        # キュー内のデータ数がサンプル数を超えている間作動
        data_queue = df_data_queue.values.tolist()
        while len(data_queue) > N:
            window = process_window(data_queue)
            if window:
                feature_list.append(get_feature.get_feature(window))
                analysis_csv.extend(label_shape(window))


# ================================= メイン関数　実行 ================================ #
if __name__ == '__main__':
    ex_num = input("実験番号：")
    do_process_window()

    # TimeStampラベルを削除
    df_analysis = pd.DataFrame(analysis_csv[1:], columns=analysis_csv[0])
    df_analysis = df_analysis.drop('timeStamp', axis=1)

    # ウィンドウ処理したデータの出力
    df_analysis.to_csv("analysis_files/window_files/window_list"+ex_num+".csv")

    # 正解データの出力
    with open("analysis_files/answer_files/answer_list"+ex_num+".csv", 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(answer_list)

    # 特徴量データの出力
    with open("analysis_files/feature_files/feature_list"+ex_num+".csv", 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(feature_list)


    # 正解データ取得
    X = pd.DataFrame(feature_list)
    y = np.loadtxt("analysis_files/answer_files/answer_list"+ex_num+".csv", delimiter=",", dtype='int')
    y = pd.Series(data=y)

    (train_x, test_x, train_y, test_y) = train_test_split(X, y, test_size=0.3)
    clf = RandomForestClassifier(max_depth=30, n_estimators=100, random_state=42)
    clf.fit(train_x, train_y)
    y_pred = clf.predict(test_x)

    print('正解データ：　{}'.format(test_y.values))
    print('予測データ：　{}'.format(y_pred))
    accuracy = accuracy_score(test_y, y_pred)
    print('Test Accuracy: {}'.format(accuracy))
    y_pred = clf.predict(train_x)
    accuracy = accuracy_score(train_y, y_pred)
    print('Train Accuracy: {}'.format(accuracy))
    print(get_feature.feature_columns)

    # ランダムフォレストの説明変数の重要度をデータフレーム化
    fea_rf_imp = pd.DataFrame({'imp': clf.feature_importances_, 'col': get_feature.feature_columns})
    fea_rf_imp = fea_rf_imp.sort_values(by='imp', ascending=False)

    # ランダムフォレストの重要度を可視化
    plt.figure(figsize=(10, 7))
    sns.barplot(x='imp', y='col', data=fea_rf_imp, orient='h')
    plt.title('Random Forest - Feature Importance', fontsize=28)
    plt.ylabel('Features', fontsize=18)
    plt.xlabel('Importance', fontsize=18)
    plt.show()

