#
# データ処理スレッド
#
import csv
import glob
from _csv import reader

import numpy as np
import collections
import statistics
import pandas as pd
import add_data
import enter_label
import stop

analysis_csv = [['action', "timeStamp", "acc_X", "acc_Y", "acc_Z", "gyro_X", "gyro_Y", "gyro_Z"]]  # windowデータの追加
answer_list = []  # 正解データリスト（windowごと）


# ============================ データ整形スレッド ============================== #
def label_shape(window):
    window_T = np.array(window).T  # 転置　（labelごとの個数を計算するため）
    label_num = collections.Counter(window_T[0])  # labelごとの個数を計算

    # labelの均一化，正解データの追加
    # if label_num['nod'] > len(window) / 2:
    #     window_name = 'nod' + str(window_num)
    #     window_T[0] = [window_name] * len(window)
    #     answer_list.append(1)
    # elif label_num['shake'] > len(window) / 2:
    #     window_name = 'shake' + str(window_num)
    #     window_T[0] = [window_name] * len(window)
    #     answer_list.append(2)
    # else:
    #     window_name = 'others' + str(window_num)
    #     window_T[0] = [window_name] * len(window)
    #     answer_list.append(0)

    if label_num['nod'] > len(window) / 2:
        answer_list.append(1)
    elif label_num['shake'] > len(window) / 2:
        answer_list.append(2)
    else:
        answer_list.append(0)
    window_T[0] = [window_num] * len(window)

    return window_T.T


# ============================ ウィンドウ処理スレッド ============================== #
# ウィンドウ単位の処理用定数
T = 10  # サンプリング周期 [Hz]
OVERLAP = 50  # オーバーラップ率 [%]
N = T * 1  # サンプル数
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
        with open(file_name, 'r') as f:
            csv_reader = reader(f)
            data_queue = list(csv_reader)

        # キュー内のデータ数がサンプル数を超えている間作動
        while len(data_queue) > N:
            window = process_window(data_queue)
            if window:
                analysis_csv.extend(label_shape(window))


# ================================= メイン関数　実行 ================================ #
if __name__ == '__main__':
    do_process_window()

    # TimeStampラベルを削除
    T_analysis_csv = np.array(analysis_csv).T
    window_T_list = T_analysis_csv.tolist()
    del window_T_list[1]
    analysis_csv = np.array(window_T_list).T

    # ウィンドウ処理したデータの出力
    pd.DataFrame(analysis_csv[1:], columns=analysis_csv[0]).to_csv('analysis_files/analysis.csv')

    # 正解データの出力
    with open('analysis_files/answer_files/answer.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(answer_list)