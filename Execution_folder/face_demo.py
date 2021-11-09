import argparse
from paz.backend import camera as FS
from paz.backend.camera import VideoPlayer
from paz.backend.camera import Camera
from paz.pipelines import detection
from paz.pipelines import DetectMiniXceptionFER
import threading
import csv
import asyncio

# 自作ライブラリ
from head_nod_analysis import client_face
from head_nod_analysis.stop import Stop
from head_nod_analysis.enter_label import Label


def face_detection():
    parser = argparse.ArgumentParser(description='Real-time face classifier')
    parser.add_argument('-c', '--camera_id', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('-o', '--offset', type=float, default=0.1,
                        help='Scaled offset to be added to bounding boxes')
    args = parser.parse_args()

    pipeline = DetectMiniXceptionFER([args.offset, args.offset])
    camera = Camera(args.camera_id)
    player = VideoPlayer((640, 480), pipeline, camera)
    player.record()


if __name__ == "__main__":
    ex_num = input('実験番号：')
    if input('サーバ通信[y/n]：') == 'y':
        server = True
    else: server = False

    # 頭の動きのセンシング　スレッド開始
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=face_detection)
    thread_1 = threading.Thread(target=client_face.client_face, args=(server,))
    thread_2 = threading.Thread(target=Stop)

    thread.start()
    thread_1.start()
    thread_2.start()
    print('start!')

    # # 顔の表情のセンシング
    # face_detection()

    # スレッドの待ち合わせ処理
    thread_list = threading.enumerate()
    thread_list.remove(threading.main_thread())
    for thread in thread_list:
        thread.join()

    print('全てのスレッドが終了しました．これからデータログを送信します．')

