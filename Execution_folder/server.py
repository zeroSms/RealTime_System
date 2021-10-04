import socket


def server():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((socket.gethostname(), 1235))  # IPとポート番号を指定します
    s.listen(5)

    while True:
        clientsocket, address = s.accept()
        print(f"Connection from {address} has been established!")
        clientsocket.send(bytes("Welcome to the server!", 'utf-8'))
        clientsocket.close()


# ================================= メイン関数　実行 ================================ #
if __name__ == '__main__':
    server()
