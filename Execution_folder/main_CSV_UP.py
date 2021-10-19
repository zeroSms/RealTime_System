#
# メイン関数
#
import os
import threading
import csv
import asyncio

# 自作ライブラリ
from head_nod_analysis import add_data, process_data, get_address, setup_variable
from head_nod_analysis.stop import Stop
from head_nod_analysis.enter_label import Label

# ================================= パスの取得 ================================ #
path = setup_variable.path


# ================================= CSV出力 ================================ #
# ログファイル出力
def getCsv_log(file_num):
    log_name = path + '/data_set/log_files/main/value_list' + file_num + '.csv'
    with open(log_name, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(add_data.log_data)


# ================================= メイン関数 ================================ #
# メイン関数
def main():
    ex_num = input('実験番号：')

    # eSenseアドレスを取得
    address = get_address.Get()

    # 頭の動きのセンシング　スレッド開始
    loop = asyncio.new_event_loop()
    thread_1 = threading.Thread(target=add_data.AddData, args=(address, loop,))
    # thread_2 = threading.Thread(target=process_data.ProcessData)
    # thread_3 = threading.Thread(target=analysis_data.AnalysisData)
    thread_4 = threading.Thread(target=Stop)
    thread_5 = threading.Thread(target=Label)

    thread_1.start()
    # thread_2.start()
    # thread_3.start()
    thread_4.start()
    thread_5.start()
    print('start!')

    # スレッドの待ち合わせ処理
    thread_list = threading.enumerate()
    thread_list.remove(threading.main_thread())
    for thread in thread_list:
        thread.join()

    print('全てのスレッドが終了しました．これからデータログを送信します．')
    getCsv_log(ex_num)  # ログファイル出力


# ================================= メイン関数　実行 ================================ #
if __name__ == '__main__':
    main()
