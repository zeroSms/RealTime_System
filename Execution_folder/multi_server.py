import socket
import pickle
import threading

# 自作ライブラリ
from head_nod_analysis import setup_variable, stop

# グローバル変数
presenter = 0
clients = []  # クライアント側の接続状況を管理
output_list, output_log = {}, {}    # 受信リスト

# 受信した文字列をJSON形式に整形
def shape_JSON(msg, address):
    # msg={'timeStamp',
    #      'class': 'Head' or 'Face'
    #      'action': 頭の動きは0~2の数値，顔の表情はa~gまでの記号
    #     }
    if address not in output_list:
        output_list[address] = {'Head': {}, 'Face': {}}
        output_log[address] = {'Head': {}, 'Face': {}}
    output_list[address][msg['class']][msg['timeStamp']] = msg['action']
    output_log[address][msg['class']][msg['timeStamp']] = msg['action']
    print(output_list)


# 接続済みクライアントは読み込みおよび書き込みを繰り返す
def loop_handler(connection, client_address):
    global presenter
    while stop.stop_flg:
        try:
            # クライアント側から受信する
            rcvmsg = connection.recv(4096)

            # 発表者デバイスのアドレスの特定
            if pickle.loads(rcvmsg) == ['host']:
                presenter = client_address[0]
                print('Presenter address -> {}'.format(presenter))
            else:
                # 受信メッセージの出力
                for value in clients:
                    if value[1][0] == client_address[0] and value[1][1] == client_address[1]:
                        print('Received {} -> {}'.format(value[1][0], pickle.loads(rcvmsg)))

                        # # 受信した文字列をJSON形式に整形
                        shape_JSON(pickle.loads(rcvmsg), client_address[0])

                # 特定のアドレスに送信
                if client_address[0] == presenter:
                    connection.sendto('aaaa'.encode('UTF-8'), client_address)  # メッセージを返します

        except Exception as e:
            print('!!!')
            print(e)
            break


def server():
    host = socket.gethostname()  # サーバーのホスト名
    port = setup_variable.port  # 49152~65535

    print(host)
    print(socket.gethostbyname(host))

    serversock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    serversock.bind((host, port))  # IPとPORTを指定してバインドします
    serversock.listen(10)  # 接続の待ち受けをします（キューの最大数を指定）

    print('Waiting for connections...')
    while stop.stop_flg:
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

        # 待受中にアクセスしてきたクライアントを追加
        clients.append((connection, client_address))
        # スレッド作成
        thread = threading.Thread(target=loop_handler, args=(connection, client_address), daemon=True)
        # スレッドスタート
        thread.start()


# ================================= メイン関数　実行 ================================ #
if __name__ == '__main__':
    # # Stopスレッド作成
    # thread_stop = threading.Thread(target=stop.Stop(), daemon=True)
    # thread_stop.start()

    # サーバーの起動
    server()

    # スレッドの待ち合わせ処理
    thread_list = threading.enumerate()
    thread_list.remove(threading.main_thread())
    for thread in thread_list:
        thread.join()
