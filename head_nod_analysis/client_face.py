#
# 表情の決定と通信処理
#
import time
import numpy as np
import collections
import pandas as pd
import socket
import pickle
import pyautogui

# 自作ライブラリ
from . import add_data, get_feature, setup_variable, stop
from paz.backend import camera as CML

# ================================= パスの取得 ================================ #
path = setup_variable.path
server_address = '192.168.2.111'


# ================================= 表情の決定・通信 ================================ #
def client_face(to_server=False):
    if to_server:
        host = server_address  # サーバーのホスト名
        client_address = socket.gethostname()  # クライアント側のホスト名
        port = setup_variable.port  # 49152~65535

        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # オブジェクトの作成をします
        client.connect((host, port))  # これでサーバーに接続します
        client.send(pickle.dumps(['host']))

        response = {'timeStamp': time.time(),
                    'class': 'Face'
                    }

    timeStamp = time.time()
    while stop.stop_flg:
        time.sleep(3)
        timeStamp = time.time()

        # 判定された表情の出力
        pred_face = CML.process_window()
        print(pred_face)

        # サーバーへの送信
        if to_server:
            response['timeStamp'] = round(timeStamp, 2)
            response['action'] = setup_variable.face_symbol(pred_face)
            massage = pickle.dumps(response)
            client.send(massage)  # データを送信

            # try:
            #     client.settimeout(2)
            #     recv_msg = client.recv(4096)  # レシーブは適当な2の累乗にします（大きすぎるとダメ）
            #     print(recv_msg.decode().replace('b', ''))
            # except Exception as e:
            #     print(e)
            #     continue
