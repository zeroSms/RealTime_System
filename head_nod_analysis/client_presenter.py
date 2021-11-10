#
# 表情の決定と通信処理
#
import time
import socket
import pickle

# 自作ライブラリ
import setup_variable

# ================================= パスの取得 ================================ #
path = setup_variable.path
server_address = '192.168.2.111'


# ================================= 表情の決定・通信 ================================ #
def client_presenter():
    host = server_address  # サーバーのホスト名
    port = setup_variable.port  # 49152~65535

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # オブジェクトの作成をします
    client.connect((host, port))  # これでサーバーに接続します

    while True:
        time.sleep(3)

        # サーバーへの送信
        response = {'presenter': True,
                    'setting': False,
                    'finish': False
                    }
        massage = pickle.dumps(response)
        client.send(massage)  # データを送信

        # データの受信
        try:
            rcvmsg = pickle.loads(client.recv(4096))
            print(rcvmsg)
        except Exception as e:
            print(e)
            continue


# ================================= メイン関数　実行 ================================ #
if __name__ == '__main__':
    client_presenter()
