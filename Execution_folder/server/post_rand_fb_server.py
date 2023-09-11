
import json
import random
import time
from pprint import pprint


N = 8  # アイコンの数
face_list = ["a", "b", "c", "e"]

# 各反応出現割合
JOY = 25
SURPRISE = 5
NOD = 15
SHAKE = 0


def post_json():
    server_json = {"server": []}    # サーバ出力用
    rand_sort = random.sample([i for i in range(N)], N)
    fb_dict = {}
    for i in range(N):
        # 頭部動作決定
        head_int = random.randint(0, 100)
        if head_int < NOD:
            set_head = 1
        elif head_int < NOD + SHAKE:
            set_head = 2
        else:
            set_head = 0

        # 表情決定
        face_int = random.randint(0, 100)
        if face_int < JOY:
            set_face = 1
        elif face_int < JOY + SURPRISE:
            set_face = 2
        else:
            set_face = 0

        fb_dict[rand_sort[i]] = {
            "id": i,
            "sort": int(rand_sort[i]),
            "Head": set_head,
            "Face": face_list[set_face]}

    for i in range(N):
        server_json["server"].append(fb_dict[i])

    # json出力
    with open("./db.json", 'w') as outfile:
        json.dump(server_json, outfile)

    print("#===== json 出力 =====#")
    pprint(server_json)


    # ============== メイン関数 実行 ============== #
if __name__ == '__main__':
    while True:
        time.sleep(7)
        post_json()
