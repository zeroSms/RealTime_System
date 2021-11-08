import socket
import pickle
import threading

# 自作ライブラリ
from head_nod_analysis import setup_variable, stop

# グローバル変数
clients = []  # クライアント側の接続状況を管理


# 接続済みクライアントは読み込みおよび書き込みを繰り返す
def loop_handler(connection, client_address):
    while True:
        try:
            # クライアント側から受信する
            rcvmsg = connection.recv(4096)
            # print(clients)
            for value in clients:
                # print(value[1][0], value[1][1])
                if value[1][0] == client_address[0] and value[1][1] == client_address[1]:
                    print('Received {} -> {}'.format(value[1][0], pickle.loads(rcvmsg)))
                # else:
                #     value[0].send(
                #         "クライアント{}:{}から{}を受信".format(value[1][0], value[1][1], rcvmsg.decode()).encode("UTF-8"))
                #     pass
            # if client_address[0] == '192.168.2.162':
            #     connection.sendto('aaaa'.encode('UTF-8'), client_address)  # メッセージを返します
            # else:   continue
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
    server()

    # スレッドの待ち合わせ処理
    thread_list = threading.enumerate()
    thread_list.remove(threading.main_thread())
    for thread in thread_list:
        thread.join()
