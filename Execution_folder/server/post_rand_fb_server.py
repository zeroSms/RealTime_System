
import json
import random
import time
from pprint import pprint


N = 24
face_list = ["a", "b", "c", "e"]


def post_json():
    server_json = {"server": []}    # サーバ出力用
    rand_sort = random.sample([i for i in range(N)], N)
    fb_dict = {}
    for i in range(N):
        fb_dict[rand_sort[i]] = {
            "id": i,
            "sort": int(rand_sort[i]),
            "Head": random.randint(0, 2),
            "Face": face_list[random.randint(0, 3)]}

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
