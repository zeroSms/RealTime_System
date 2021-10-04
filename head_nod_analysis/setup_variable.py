#
# データ取得スレッド
#


# ============================ 変数宣言部 ============================== #
import os
path = os.getcwd().rsplit('\\', 1)[0]

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

# ウィンドウ単位の処理用定数
T = 100  # サンプリング周期 [Hz]
N = 32  # ウィンドウサイズ(0.64秒)
OVERLAP = 50  # オーバーラップ率 [%]
FOLD = 10
