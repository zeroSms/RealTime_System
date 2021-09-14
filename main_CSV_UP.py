#
# メイン関数
#

import threading
import add_data
import process_data
import get_address
import csv
import asyncio
# import analysis_data
import pandas as pd
from stop import Stop
from enter_label import Label

# ================================= CSV出力 ================================ #
# ログファイル出力
def getCsv_log(file_num):
    log_name = "log_files/value_list" + file_num + ".csv"
    with open(log_name, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(add_data.log_data)

# 教師データ，正解データ出力
def getCsv_analy(ex_num):
    window_name = "analysis_files/window_files/window_list" + ex_num + ".csv"
    answer_name = "analysis_files/answer_files/answer_list" + ex_num + ".csv"
    print(process_data.answer_list)
    with open(window_name, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(process_data.analysis_csv)
    with open(answer_name, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(process_data.answer_list)


# ================================= メイン関数 ================================ #
# メイン関数
def main():
    file_num = input("file_number 入力：")
    ex_num = input("実験番号：")

    address = get_address.Get()

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
    print("start!")

    # スレッドの待ち合わせ処理
    thread_list = threading.enumerate()
    thread_list.remove(threading.main_thread())
    for thread in thread_list:
        thread.join()

    print("全てのスレッドが終了しました．これからデータログを送信します．")

    getCsv_log(file_num)        # ログファイル出力
    # getCsv_analy(ex_num)        # 教師データ，正解データ出力


# ================================= メイン関数　実行 ================================ #
if __name__ == '__main__':
    main()
