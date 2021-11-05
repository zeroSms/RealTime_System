# ライブラリ取得
import socket
import pickle
import time

# ================================= パスの取得 ================================ #
server_address = '192.168.2.111'

def client():
    host = server_address  # サーバーのホスト名
    port = 50000  # 49152~65535

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # オブジェクトの作成をします
    client.connect((host, port))  # これでサーバーに接続します

    while True:
        massage = pickle.dumps({'timeStamp': time.time(),
                                'iD': host,



        })
        client.send(massage)  # 適当なデータを送信します（届く側にわかるように）

        time.sleep(1)

        # response = client.recv(4096)  # レシーブは適当な2の累乗にします（大きすぎるとダメ）
        #
        # print(str(response).replace('b', ''))
    # client.close()


# ================================= メイン関数　実行 ================================ #
if __name__ == '__main__':
    client()
