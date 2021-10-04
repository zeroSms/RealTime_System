import socket


def client():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((socket.gethostname(), 1235))
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    msg = s.recv(1024)
    print(msg.decode("utf-8"))


# ================================= メイン関数　実行 ================================ #
if __name__ == '__main__':
    client()
