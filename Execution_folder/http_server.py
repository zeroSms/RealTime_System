import pickle
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse
import json
import collections
import socket

# 自作ライブラリ
from head_nod_analysis import setup_variable, stop

# ================================= パスの取得 ================================ #
server_address = setup_variable.server_address  # '192.168.2.111'
presenter_port = setup_variable.presenter_port  # 5000
audience_port = setup_variable.audience_port  # 50000

# サーバーへの送信
response = {'presenter': True,
            'setting': False,
            'finish': False
            }

def connect_socket():
    host = server_address  # サーバーのホスト名
    port = setup_variable.audience_port  # 49152~65535

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # オブジェクトの作成をします
    client.connect((host, port))  # これでサーバーに接続します

    massage = pickle.dumps(response)
    client.send(massage)  # データを送信

    # 集約データを受信
    try:
        recv_msg = client.recv(4096)
        print(pickle.loads(recv_msg))
    except Exception as e:
        print(e)

    return recv_msg


# ================================= サーバ処理 ================================ #
class MyHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        print('path = {}'.format(self.path))

        parsed_path = urlparse(self.path)
        print('parsed: path = {}, query = {}'.format(parsed_path.path, parse_qs(parsed_path.query)))

        print('headers\r\n-----\r\n{}-----'.format(self.headers))

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(b'Hello from do_GET')

    def do_POST(self):
        print('path = {}'.format(self.path))

        parsed_path = urlparse(self.path)
        print('parsed: path = {}, query = {}'.format(parsed_path.path, parse_qs(parsed_path.query)))

        print('headers\r\n-----\r\n{}-----'.format(self.headers))

        content_length = int(self.headers['content-length'])

        rcvmsg = self.rfile.read(content_length).decode('utf-8')
        print('body = {}'.format(rcvmsg))

        # 視聴者側サーバとの送受信
        send_msg = connect_socket()

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(send_msg)


with HTTPServer((server_address, presenter_port), MyHTTPRequestHandler) as server:
    server.serve_forever()
