#
# メイン関数
#

import threading
import add_data
import csv
import asyncio

import analysis_data
import pandas as pd
from stop import Stop


# ================================= メイン関数 ================================ #

# メイン関数
def main():
    filename = "data_files/others/value_list" + input("file_number 入力：") + ".csv"
    print("ファイル名：" + filename)

    address = add_data.get_address()

    print("aa")
    thread_1 = threading.Thread(target=add_data.AddData(address))
    # thread_2 = threading.Thread(target=ProcessData)
    # thread_3 = threading.Thread(target=analysis_data.AnalysisData)
    thread_4 = threading.Thread(target=Stop)

    print("start!")
    thread_1.start()
    # thread_2.start()
    # thread_3.start()
    thread_4.start()

    # スレッドの待ち合わせ処理
    thread_list = threading.enumerate()
    thread_list.remove(threading.main_thread())
    for thread in thread_list:
        thread.join()

    print("全てのスレッドが終了しました．これからデータログを送信します．")

    with open(filename, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(add_data.data_queue)

# ================================= メイン関数　実行 ================================ #
if __name__ == '__main__':
    main()
