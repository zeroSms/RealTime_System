#
# データ取得スレッド
#


# ============================ 変数宣言部 ============================== #
# 分析用データのラベル

process_columns = ['window_ID', 'timeStamp',
                   'acc_X', 'acc_Y', 'acc_Z',
                   'gyro_X', 'gyro_Y', 'gyro_Z']

analysis_columns = ['window_ID', 'timeStamp',
                    'acc_X', 'acc_Y', 'acc_Z',
                    'gyro_X', 'gyro_Y', 'gyro_Z',
                    'acc_xyz', 'gyro_xyz']

axis_columns = ['acc_X', 'acc_Y', 'acc_Z',
                'gyro_X', 'gyro_Y', 'gyro_Z',
                'acc_xyz', 'gyro_xyz']

# ウィンドウ単位の処理用定数
T = 50  # サンプリング周期 [Hz]
N = 64  # ウィンドウサイズ(0.781秒)
OVERLAP = 50  # オーバーラップ率 [%]
