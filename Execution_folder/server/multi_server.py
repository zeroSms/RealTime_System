import collections
import copy
import socket
import pickle
import sys
import threading
import json
import time
import datetime

# 自作ライブラリ
from head_nod_analysis import setup_variable


# ============================ 正常終了スレッド ============================== #se
# プログラムを正常に終了するための関数
def Stop(server_num):
    # エンターが入力されるまで待ち続ける
    input()
    print("stop!\n")

    # ログファイルの出力
    with open('../../server_files/server_log/' + server_num + '.json', 'w') as f:
        json.dump(output_log, f, indent=4)
    sys.exit()


# ============================ マルチサーバー ============================== #
# グローバル変数
clients = []  # クライアント側の接続状況を管理
output_list, output_log = {}, {}  # 受信リスト


# 受信した文字列をJSON形式に整形
def shape_JSON(msg):
    global output_list, output_log
    """
    msg = { 'presenter': True or False,
            'ID': 1-8
            'timeStamp',
            'class': 'Head' or 'Face',
            'action': 頭の動きは0~2の数値，顔の表情はa~gまでの記号
           }

    output = {'ID':{'Head': {timeStamp: action},
                    'Face': {timeStamp: action}
                    }
                          ...
              }
    """
    if msg['ID'] not in output_list:
        output_list[msg['ID']] = {'Head': {}, 'Face': {}}
        # output_log[address] = {'Head': {}, 'Face': {}}
    output_list[msg['ID']][msg['class']][str(time.time())] = msg['action']
    # output_log[address][msg['class']][str(time.time())] = msg['action']


def to_presenter(presenter_address, connection):
    """
    msg = { 'presenter': True,
            'timeStamp': round(time.time(), 2)
            'finish': True or False
           }

    to_list = {'Sum': N,
               'audience': {'ID': {'Head': 'action', 'Face': 'action'},
                            'ID': {'Head': 'action', 'Face': 'action'},
                                    ...
                            }
               }
    """
    to_list = {}  # 送信用リスト
    next_check_list = {}  # 次回繰り越し用リスト（要素数3未満の場合）

    # output_listの複製・初期化
    global output_list
    output_copy = copy.deepcopy(output_list)
    output_list = {}

    # 各ユーザの頭の動きの決定
    def to_head(ID):
        head_list = list(output_copy[ID]['Head'].values())
        # 前回繰り越しリストがあれば連結
        if ID in next_check_list:
            head_list = next_check_list[ID] + head_list
            del next_check_list[ID]

        # 要素数3未満なら次回に繰り越し
        if len(head_list) < 3:
            next_check_list[ID] = copy.copy(head_list)

        # 返す値の決定
        count_1 = head_list.count(1)
        count_2 = head_list.count(2)
        if count_1 == count_2 and count_1 > 0:
            return 3
        elif count_1 > count_2:
            return 1
        elif count_2 > count_1:
            return 2
        else:
            return 0

    # 各ユーザの表情の決定
    def to_face(ID):
        face_list = list(output_copy[ID]['Face'].values())
        max_num = 0.0
        max_face = 'z'
        for face_score in face_list:
            if face_score != 'null' and face_score[0] not in ["neutral", "null"] and face_score[1] > max_num:
                max_face = face_score[0]
                max_num = face_score[1]

        if max_face == 'z':
            return 'z'
        else:
            return setup_variable.face_symbol(max_face)

        # if max_face == 'z':
        #     return 'z'
        # if max_num > 0.3:
        #     return setup_variable.face_symbol(max_face)
        # else:
        #     return 'a'

    # 視聴者の人数を計算
    audience = len(output_copy)
    if audience > 0:
        # to_listの初期化
        to_list['Sum'] = audience
        to_list['audience'] = {}

        for ID in output_copy.keys():
            to_list['audience'][ID] = {}

            # 発表者へ送ったフィードバック内容の記録
            if ID not in output_log:
                output_log[ID] = {'Head': {}, 'Face': {}}

            # すべての反応をフィードバック
            if output_copy[ID]['Head']:
                to_list['audience'][ID]['Head'] = to_head(ID)
            else:
                to_list['audience'][ID]['Head'] = 0
            if output_copy[ID]['Face']:
                to_list['audience'][ID]['Face'] = to_face(ID)
            else:
                to_list['audience'][ID]['Face'] = 'z'

            timeStamp = str(time.time())
            output_log[ID]['Head'][timeStamp] = to_list['audience'][ID]['Head']
            output_log[ID]['Face'][timeStamp] = to_list['audience'][ID]['Face']

        print(to_list)
        server_json = {"server": [to_list]}
        # connection.sendto(pickle.dumps(to_list), presenter_address)  # メッセージを返します
        with open("./db.json", 'w') as outfile:
            json.dump(server_json, outfile)



# 接続済みクライアントは読み込みおよび書き込みを繰り返す
def loop_handler(connection, client_address):
    start_time = datetime.datetime.now()
    while True:
        # 一定時間ごとにJSONファイルを更新
        now_time = datetime.datetime.now()
        if now_time > start_time + datetime.timedelta(seconds=7):
            start_time = now_time
            print(output_list)
            if len(output_list) != 0:
                print("OK")
                to_presenter(client_address, connection)
            else:
                print("NG")

        try:
            # クライアント側から受信する
            rcvmsg = connection.recv(4096)
            if type(rcvmsg) is bytes:
                rcvmsg = pickle.loads(rcvmsg)

            # 発表者デバイスとの送受信
            if rcvmsg['presenter'] == True:
                # 切断処理
                if rcvmsg['finish'] == True:
                    print('発表者デバイスとの通信を終了')
                    break

                # 発表デバイスに送信
                to_presenter(client_address, connection)

            # 視聴者デバイスからの受信
            else:
                # 受信メッセージの出力
                for value in clients:
                    if value[1][0] == client_address[0] and value[1][1] == client_address[1]:
                        print('Received {} -> {}'.format(value[1][0], rcvmsg))

                        # 受信した文字列をJSON形式に整形/output_listに追加
                        # shape_JSON(rcvmsg, client_address[0])
                        shape_JSON(rcvmsg)

        except Exception as e:
            print('!!!')
            print(e)
            break

def server():
    host = socket.gethostname()  # サーバーのホスト名
    port = setup_variable.port_num[port_select]['audience']

    print(host)
    print(socket.gethostbyname(host))

    serversock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    serversock.bind((host, port))  # IPとPORTを指定してバインドします
    serversock.listen(10)  # 接続の待ち受けをします（キューの最大数を指定）

    print('Waiting for connections...')
    while True:
        try:
            # 接続要求を受信
            connection, client_address = serversock.accept()  # 接続されればデータを格納

        except KeyboardInterrupt:
            serversock.close()
            print('???')
            exit()
            break

        # アドレス確認
        print("\r\n")
        print("[アクセス元アドレス]=>{}".format(client_address[0]))
        print("[アクセス元ポート]=>{}".format(client_address[1]))
        print(connection)

        # 待受中にアクセスしてきたクライアントを追加
        clients.append((connection, client_address))
        # スレッド作成
        thread = threading.Thread(target=loop_handler, args=(connection, client_address), daemon=True)
        # スレッドスタート
        thread.start()


# ================================= メイン関数　実行 ================================ #
if __name__ == '__main__':
    server_num = input('サーバー番号：')
    port_select = input('ポート番号[1/2/3]：')

    # サーバーの起動
    thread_stop = threading.Thread(target=server, daemon=True)
    thread_stop.start()

    # システムの終了関数/ログファイルの出力
    Stop(server_num)

    # スレッドの待ち合わせ処理
    thread_list = threading.enumerate()
    thread_list.remove(threading.main_thread())
    for thread in thread_list:
        thread.join()
