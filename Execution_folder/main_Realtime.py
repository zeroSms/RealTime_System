#
# メイン関数
#
import threading
import csv
import asyncio
import os
import shutil

# 自作ライブラリ
from head_nod_analysis import add_data, process_data, get_address, setup_variable
from head_nod_analysis.enter_label import Label
from head_nod_analysis.stop import Stop

# ================================= パスの取得 ================================ #
path = setup_variable.path

# ================================= CSV出力 ================================ #
# ログファイル出力
def getCsv_log(realtime_file, ex_num):
    log_name = realtime_file + '\\value_list' + ex_num + '.csv'
    with open(log_name, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(add_data.log_data)

# 教師データ，正解データ出力
def getCsv_analysis(realtime_file, ex_num):
    window_name = realtime_file + '\\window_list' + ex_num + '.csv'
    answer_name = realtime_file + '\\answer_list' + ex_num + '.csv'
    with open(window_name, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(process_data.analysis_csv)
    with open(answer_name, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(process_data.answer_list)


# ================================= メイン関数 ================================ #
# メイン関数
def main():
    ex_num = input('実験番号：')
    port_select = input('ポート番号[1/2/3]：')
    if input('サーバ通信[y/n]：') == 'y':
        server = True
    else: server = False

    # eSenseのアドレスを取得
    address = get_address.Get()

    loop = asyncio.new_event_loop()
    thread_1 = threading.Thread(target=add_data.AddData, args=(address, loop,))
    thread_2 = threading.Thread(target=process_data.Realtime_analysis, args=(server, port_select,))
    thread_3 = threading.Thread(target=Stop)
    thread_4 = threading.Thread(target=Label)

    thread_1.start()
    thread_2.start()
    thread_3.start()
    thread_4.start()
    print('start!')

    # スレッドの待ち合わせ処理
    thread_list = threading.enumerate()
    thread_list.remove(threading.main_thread())
    for thread in thread_list:
        thread.join()

    print('全てのスレッドが終了しました．これからデータログを送信します．')

    # realtime_filesの初期化
    realtime_file = path + '\\data_set\\realtime_files\\' + str(ex_num)
    if os.path.exists(realtime_file):
        shutil.rmtree(realtime_file)
    os.makedirs(realtime_file)

    getCsv_log(realtime_file, ex_num)        # ログファイル出力
    getCsv_analysis(realtime_file, ex_num)        # 教師データ，正解データ出力

    with open(realtime_file + '\\predict_list' + ex_num + '.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(process_data.realtime_pred)


# ================================= メイン関数　実行 ================================ #
if __name__ == '__main__':
    main()
