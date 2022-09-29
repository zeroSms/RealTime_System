import pickle
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse
import json
import collections
import socket
import time

# 自作ライブラリ
from head_nod_analysis import setup_variable, stop

# ================================= パスの取得 ================================ #
server_address = setup_variable.server_address  # '192.168.2.111'

# サーバーへの送信
response = {'presenter': True,
            'timeStamp': round(time.time(), 2),
            'finish': False
            }


def connect_socket():
    host = server_address  # サーバーのホスト名
    port = audience_port

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # オブジェクトの作成をします
    client.connect((host, port))  # これでサーバーに接続します

    print(response)

    massage = pickle.dumps(response)
    client.send(massage)  # データを送信

    # 集約データを受信
    try:
        send_msg = pickle.loads(client.recv(4096))
        print(send_msg)
    except Exception as e:
        print(e)

    return send_msg


# ================================= サーバ処理 ================================ #
class MyHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With, Content-Type")
        self.end_headers()

    def do_GET(self):
        print('path = {}'.format(self.path))

        parsed_path = urlparse(self.path)
        print('parsed: path = {}, query = {}'.format(parsed_path.path, parse_qs(parsed_path.query)))

        print('headers\r\n-----\r\n{}-----'.format(self.headers))
        print("bbbbb")

        # self.send_response(200)
        # self.send_header('Content-Type', 'application/json')
        # self.end_headers()
        # self.wfile.write(b'Hello from do_GET')

        # 視聴者側サーバとの送受信
        send_msg = connect_socket()

        self.send_response(800)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(send_msg).encode())

    def do_POST(self):
        print('path = {}'.format(self.path))

        parsed_path = urlparse(self.path)
        print('parsed: path = {}, query = {}'.format(parsed_path.path, parse_qs(parsed_path.query)))

        print('headers\r\n-----\r\n{}-----'.format(self.headers))
        print("aaaaa")

        content_length = int(self.headers['content-length'])

        rcvmsg = self.rfile.read(content_length).decode('utf-8')
        rcvmsg = json.loads(rcvmsg)
        print('body = {}'.format(rcvmsg))

        # 視聴者側サーバとの送受信
        send_msg = connect_socket()

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(send_msg).encode())


# ================================= メイン関数　実行 ================================ #
if __name__ == '__main__':
    port_select = input('ポート番号[1/2/3]：')
    presenter_port = setup_variable.port_num[port_select]['presenter']
    audience_port = setup_variable.port_num[port_select]['audience']

    with HTTPServer((server_address, presenter_port), MyHTTPRequestHandler) as server:
        server.serve_forever()
