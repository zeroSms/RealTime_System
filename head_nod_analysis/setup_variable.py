#
# データ取得スレッド
#


# ================================= パスの取得 ================================ #
import os

path = os.getcwd().rsplit('\\', 1)[0]
server_address = '192.168.2.111'
# server_address = '192.168.2.22'  # 3A, notePC
# server_address = '172.19.0.44'  # cs-wlan, notePC
# server_address = '192.168.0.2'  # 4H-aterm, notePC

port_num = {'1': {'audience': 50010}}

eSense_name = ['eSense-0220', 'eSense-0328', 'eSense-0376', 'eSense-0459', 'eSense-0498',            
               'eSense-0586', 'eSense-0592', 'eSense-0666', 'eSense-0634', 'eSense-0740']

# ============================ 変数宣言部 ============================== #
# 分析用データのラベル
process_columns = ['window_ID', 'timeStamp',
                   'acc_X', 'acc_Y', 'acc_Z',
                   'gyro_X', 'gyro_Y', 'gyro_Z']

analysis_columns = ['window_ID', 'timeStamp',
                    'acc_X', 'acc_Y', 'acc_Z',
                    'gyro_X', 'gyro_Y', 'gyro_Z',
                    'acc_xyz']

axis_columns = ['acc_X', 'acc_Y', 'acc_Z',
                'gyro_X', 'gyro_Y', 'gyro_Z',
                'acc_xyz']
acc_columns = ['acc_X', 'acc_Y', 'acc_Z', 'acc_xyz']
gyro_columns = ['gyro_X', 'gyro_Y', 'gyro_Z']


# 表情の文字列を記号に変換
def face_symbol(pred_face):
    face_dict = {
        'neutral': 'a',
        'happy': 'b',
        'surprise': 'c',
        'sad': 'd',
        'angry': 'e',
        'fear': 'f',
        'disgust': 'g',
        'null': 'z'
    }
    return face_dict[pred_face]


# ============================ ウィンドウ単位の処理用定数 ============================== #
T = 20  # サンプリング周期 [Hz]
conn = 70  # 通信間隔 [ms]
N = 32  # ウィンドウサイズ
OVERLAP = 50  # オーバーラップ率 [%]
threshold = 0.2  # ウィンドウラベル閾値

# ============================ 精度検証用定数 ============================== #
FOLD = 10  # 交差検証数

# ランダムフォレストパラメータ
random_state = 42
