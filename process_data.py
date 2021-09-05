#
# データ処理スレッド
#

import numpy as np
import collections
import statistics
import add_data
import enter_label
import stop

analysis_csv = [['action', "timeStamp", "acc_X", "acc_Y", "acc_Z", "gyro_X", "gyro_Y", "gyro_Z"]]   # windowデータの追加
answer_list = []    # 正解データリスト（windowごと）

# ============================ データ処理スレッド ============================== #
def label_shape(window, window_num):
    window_T = np.array(window).T  # 転置　（labelごとの個数を計算するため）
    label_num = collections.Counter(window_T[0])  # labelごとの個数を計算

    # labelの均一化，正解データの追加
    if label_num['nod'] > len(window) / 2:
        window_name = 'nod' + str(window_num)
        window_T[0] = [window_name] * len(window)
        answer_list.append(1)
    elif label_num['shake'] > len(window) / 2:
        window_name = 'shake' + str(window_num)
        window_T[0] = [window_name] * len(window)
        answer_list.append(2)
    else:
        window_name = 'others' + str(window_num)
        window_T[0] = [window_name] * len(window)
        answer_list.append(0)

    return window_T.T

# 測定したデータを処理する関数
def ProcessData():
    while stop.stop_flg:
        window, window_num = add_data.sensor.process_window()
        if window:
            analysis_csv.extend(label_shape(window, window_num))







