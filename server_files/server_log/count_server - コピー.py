# 特徴量リストの出力
import collections
import glob
import json

save_count = {}


def count_json(df, file_name):
    for user in df:
        save_count[file_name][user] = {}
        if 'Head' in df[user]:
            save_count[file_name][user]['Head'] = {}
            to_list_head = list(df[user]['Head'].values())
            count_head = collections.Counter(to_list_head)
            # 頭の動きを加算
            for action, num in count_head.items():
                if action not in save_count[file_name][user]['Head']:
                    save_count[file_name][user]['Head'][action] = num
                else:
                    save_count[file_name][user]['Head'][action] += num

        if 'Face' in df[user]:
            save_count[file_name][user]['Face'] = {}
            to_list_face = list(df[user]['Face'].values())
            count_face = collections.Counter(to_list_face)
            # 表情を加算
            for action, num in count_face.items():
                if action not in save_count[file_name][user]['Face']:
                    save_count[file_name][user]['Face'][action] = num
                else:
                    save_count[file_name][user]['Face'][action] += num


# ================================= メイン関数　実行 ================================ #
if __name__ == '__main__':
    server_file = input('実験方法[C/D]：')

    glob_file = server_file + '\\*.json'  # 全ファイル
    log_list = glob.glob(glob_file)
    for file_name in log_list:
        save_count[file_name] = {}
        with open(file_name) as f:
            df = json.load(f)
        count_json(df, file_name)

    # for head_nod in save_count:
    #     sum_count = sum(save_count[head_nod].values())
    #     for action in save_count[head_nod]:
    #         save_count[head_nod][action] /= sum_count


    # ログファイルの出力
    with open(server_file + '\\action_count_user.json', 'w') as f:
        json.dump(save_count, f, indent=4)
