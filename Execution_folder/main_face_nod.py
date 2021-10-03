#
# メイン関数
#
import threading
import csv
import asyncio

# 自作ライブラリ
from head_nod_analysis import add_data, process_data, get_address, setup_variable
from head_nod_analysis.stop import Stop
from head_nod_analysis.enter_label import Label
import face_demo

# ================================= パスの取得 ================================ #
path = setup_variable.path

# ================================= CSV出力 ================================ #
# ログファイル出力
def getCsv_log(ex_num):
    log_name = path + '/data_set/realtime_files/log_files/value_list' + ex_num + '.csv'
    with open(log_name, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(add_data.log_data)

# 教師データ，正解データ出力
def getCsv_analysis(ex_num):
    window_name = path + '/data_set/realtime_files/window_files/window_list' + ex_num + '.csv'
    answer_name = path + '/data_set/realtime_files/answer_files/answer_list' + ex_num + '.csv'
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
    ex_num = input('実験番号：')

    # eSenseのアドレスを取得
    address = get_address.Get()

    loop = asyncio.new_event_loop()
    thread_1 = threading.Thread(target=add_data.AddData, args=(address, loop,))
    thread_2 = threading.Thread(target=process_data.Realtime_analysis)
    # thread_3 = threading.Thread(target=face_demo.face_detection())
    thread_4 = threading.Thread(target=Stop)
    thread_5 = threading.Thread(target=Label)

    thread_1.start()
    thread_2.start()
    # thread_3.start()
    thread_4.start()
    thread_5.start()
    print('start!')

    face_demo.face_detection()

    # スレッドの待ち合わせ処理
    thread_list = threading.enumerate()
    thread_list.remove(threading.main_thread())
    for thread in thread_list:
        thread.join()

    print('全てのスレッドが終了しました．これからデータログを送信します．')

    getCsv_log(ex_num)        # ログファイル出力
    getCsv_analysis(ex_num)        # 教師データ，正解データ出力

    with open(path + '/data_set/realtime_files/predict_files/predict_list' + ex_num + '.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(process_data.realtime_pred)


# ================================= メイン関数　実行 ================================ #
if __name__ == '__main__':
    main()
