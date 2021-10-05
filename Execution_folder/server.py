import socket
import pickle


def server():
    host = socket.gethostname()  # サーバーのホスト名
    port = 50000  # 49152~65535

    print(host)
    print(socket.gethostbyname(host))

    serversock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    serversock.bind((host, port))  # IPとPORTを指定してバインドします
    serversock.listen(10)  # 接続の待ち受けをします（キューの最大数を指定）

    print('Waiting for connections...')
    clientsock, client_address = serversock.accept()  # 接続されればデータを格納

    while True:
        rcvmsg = clientsock.recv(1024)
        print('Received -> %s' % pickle.loads(rcvmsg))
        if rcvmsg == b'':
            break
        # print('Type message...')
        # s_msg = input().encode('utf-8')
        # print(s_msg)
        # if s_msg == b'':
        #     break
        # print('Wait...')
        # clientsock.sendall(s_msg)  # メッセージを返します
    clientsock.close()


# ================================= メイン関数　実行 ================================ #
if __name__ == '__main__':
    server()
