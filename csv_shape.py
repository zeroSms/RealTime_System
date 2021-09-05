import glob
import re

import pandas as pd
import csv

# CSVファイル抽出
file_list = glob.glob('analysis_files/data_files/value_list*.csv')

def push_filename():
    for file_name in file_list:
        df = pd.read_csv(file_name, encoding='utf-8',
                         names=['action', 'timeStamp', 'acc_X', 'acc_Y', 'acc_Z', 'gyro_X', 'gyro_Y', 'gyro_Z'])
        df['filename'] = re.split('[/.]', file_name)[2]
        df.to_csv(file_name, index=False, encoding='utf-8',
                  columns=['filename', 'action', 'timeStamp', 'acc_X', 'acc_Y', 'acc_Z', 'gyro_X', 'gyro_Y', 'gyro_Z'])

# answer_list = []
# i = 0
# for file_name in file_list:
#     i += 1
#     df = pd.read_csv(file_name, encoding='utf-8',
#                      names=['timeStamp', 'acc_X', 'acc_Y', 'acc_Z', 'gyro_X', 'gyro_Y', 'gyro_Z'])
#     if file_list == 'nod_file': df['action'] = 'nod' + str(i)
#     elif file_list == 'shake_file': df['action'] = 'shake' + str(i)
#     else: df['action'] = 'others' + str(i)
#     df.to_csv(file_name, index=False, encoding='utf-8',
#               columns=['action', 'timeStamp', 'acc_X', 'acc_Y', 'acc_Z', 'gyro_X', 'gyro_Y', 'gyro_Z'])
# 
#     if file_list == 'nod_file': answer_list.append(0)       # うなずき
#     elif file_list == 'shake_file': answer_list.append(1)   # 首振り
#     else: answer_list.append(2)                             # その他


# df_all_files = []
# for file_list in file_dict:
#     for file_name in file_dict[file_list]:
#         df = pd.read_csv(file_name, encoding='utf-8')
#         df_all_files.append(df)
# analysis_file = pd.concat(df_all_files, ignore_index=True)
# analysis_file.to_csv('analysis_file.csv')
# 
# with open('analysis_file_answer.csv', 'w') as f:
#     writer = csv.writer(f, lineterminator='\n')
#     writer.writerow(answer_list)