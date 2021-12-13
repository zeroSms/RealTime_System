import argparse
import asyncio
import threading

# 自作ライブラリ
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from head_nod_analysis import client_face
from head_nod_analysis.stop import Stop

from paz.backend.camera import Camera
from paz.backend.camera import VideoPlayer
from paz.pipelines import DetectMiniXceptionFER


def face_detection(ex_num):
    parser = argparse.ArgumentParser(description='Real-time face classifier')
    parser.add_argument('-c', '--camera_id', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('-o', '--offset', type=float, default=0.1,
                        help='Scaled offset to be added to bounding boxes')
    args = parser.parse_args()

    pipeline = DetectMiniXceptionFER([args.offset, args.offset])
    camera = Camera(args.camera_id)
    player = VideoPlayer((640, 480), pipeline, camera, ex_num)
    player.record()


if __name__ == "__main__":
    ex_num = input('実験番号：')
    if input('サーバ通信[y/n]：') == 'y':
        port_select = input('サーバ番号[1/2/3]：')
        server = True
    else:
        port_select = '1'
        server = False

    # 頭の動きのセンシング　スレッド開始
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=face_detection, args=(ex_num,))
    thread_1 = threading.Thread(target=client_face.client_face, args=(server, port_select,))
    thread_2 = threading.Thread(target=Stop)

    thread.start()
    thread_1.start()
    thread_2.start()
    print('start!')

    # スレッドの待ち合わせ処理
    thread_list = threading.enumerate()
    thread_list.remove(threading.main_thread())
    for thread in thread_list:
        thread.join()

    print('全てのスレッドが終了しました．これからデータログを送信します．')

