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


# ================================= メイン関数 ================================ #

# メイン関数
def main():
    filename = "data_files/others/value_list" + input("file_number 入力：") + ".csv"
    print("ファイル名：" + filename)
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

    with open(filename, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(add_data.data_list)

    # window_name = "window_files/window_list" + ex_num + ".csv"
    # answer_name = "answer_files/answer_list" + ex_num + ".csv"
    # print(process_data.answer_list)
    # with open(window_name, 'w') as f:
    #     writer = csv.writer(f, lineterminator='\n')
    #     writer.writerows(process_data.analysis_csv)
    # with open(answer_name, 'w') as f:
    #     writer = csv.writer(f, lineterminator='\n')
    #     writer.writerow(process_data.answer_list)


# ================================= メイン関数　実行 ================================ #
if __name__ == '__main__':
    main()
