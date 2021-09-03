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

# ウィンドウ処理を行う
T = 10  # サンプリング周期 [Hz]
OVERLAP = 50  # オーバーラップ率 [%]
N = T * 2  # サンプル数
byte_sample = bytearray([0x53, 0x03, 0x02, 0x01, 0x00])  # UUID7　書き込み用バイト（サンプリング開始）
eSense_address = 0
window_num = 0  # window番号
w = []
data_queue = []  # 保存用変数
data_list = []
sensor = 0
window = []
def process_window():
    global window_num, w, window
    # キュー内のデータ数がサンプル数を超えたら作動
    if len(add_data.data_queue) > N:
        not_dup = int(N * (1 - OVERLAP / 100))  # 重複しない部分の個数
        if not_dup < 1:
            not_dup = 1

        # サンプル数（N）分のデータを格納するリスト（window）の作成
        for _ in range(not_dup):
            # 重複しない部分のデータはキューから削除
            window.append(add_data.data_queue.pop(0))
        for i in range(N - not_dup):
            window.append(add_data.data_queue[i])

        if window:
            w = window
            window = []  # ウィンドウをリセット
            window_num += 1

# ============================ データ処理スレッド ============================== #
def label_shape():
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
        process_window()
        if window:
            analysis_csv.extend(label_shape())







