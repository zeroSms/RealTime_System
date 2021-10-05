import socket
import pickle


def client():
    host = '192.168.2.111'  # サーバーのホスト名
    print(host)
    print(socket.gethostbyname(host))
    port = 50000  # 49152~65535

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # オブジェクトの作成をします
    client.connect((host, port))  # これでサーバーに接続します

    while True:
        massage = pickle.dumps(["11", "22"])

        client.send(massage)  # 適当なデータを送信します（届く側にわかるように）

        # response = client.recv(4096)  # レシーブは適当な2の累乗にします（大きすぎるとダメ）
        #
        # print(str(response).replace('b', ''))


# ================================= メイン関数　実行 ================================ #
if __name__ == '__main__':
    client()
