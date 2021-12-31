# 特徴量リストの出力
import collections
import glob
import json

save_count = {'Head': {},
              'Face': {}}


def count_json(df):
    for user in df:
        if 'Head' in df[user]:
            to_list_head = list(df[user]['Head'].values())
            count_head = collections.Counter(to_list_head)
            # 頭の動きを加算
            for action, num in count_head.items():
                if action not in save_count['Head']:
                    save_count['Head'][action] = num
                else:
                    save_count['Head'][action] += num

        if 'Face' in df[user]:
            to_list_face = list(df[user]['Face'].values())
            count_face = collections.Counter(to_list_face)
            # 表情を加算
            for action, num in count_face.items():
                if action not in save_count['Face']:
                    save_count['Face'][action] = num
                else:
                    save_count['Face'][action] += num


# ================================= メイン関数　実行 ================================ #
if __name__ == '__main__':
    server_file = input('実験方法[C/D]：')

    glob_file = server_file + '\\*.json'  # 全ファイル
    log_list = glob.glob(glob_file)
    for file_name in log_list:
        with open(file_name) as f:
            df = json.load(f)
        count_json(df)

    for head_nod in save_count:
        sum_count = sum(save_count[head_nod].values())
        for action in save_count[head_nod]:
            save_count[head_nod][action] /= sum_count


    # ログファイルの出力
    with open(server_file + '\\action_count.json', 'w') as f:
        json.dump(save_count, f, indent=4)
