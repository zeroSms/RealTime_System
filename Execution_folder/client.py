import socket


def client():

    host = socket.gethostname()  # お使いのサーバーのホスト名を入れます
    port = 8888  # 適当なPORTを指定してあげます

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # オブジェクトの作成をします

    client.connect((host, port))  # これでサーバーに接続します

    massage = "from nadechin"

    client.send(massage.encode('utf-8'))  # 適当なデータを送信します（届く側にわかるように）

    response = client.recv(4096)  # レシーブは適当な2の累乗にします（大きすぎるとダメ）

    print(response)


# ================================= メイン関数　実行 ================================ #
if __name__ == '__main__':
    client()
