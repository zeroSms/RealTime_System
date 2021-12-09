#
# 収集データの可視化
#
import glob
import pandas as pd
import matplotlib.pyplot as plt

# 自作ライブラリ
from head_nod_analysis import setup_variable

# ============================ 可視化部 ============================== #
# 分析用データのラベル
number = 0

def set_plt_acc(df):
    global number
    plt.figure()
    ax = df.plot(x='time', y='acc_X',
                 title='Acceleration',
                 colormap='Accent',
                 linewidth=0.6,
                 figsize=(20, 6)
                 )
    df.plot(x='time', y='acc_Y', ax=ax,
            linewidth=0.6)
    df.plot(x='time', y='acc_Z', ax=ax,
            linewidth=0.6)
    ax.grid(True)
    ax.set_xlabel('time[s]')
    ax.set_ylabel('[m/s^2]')

    plt.savefig('fig/'+ str(number) + '.png')
    plt.close('all')
    number += 1

def set_plt_gyro(df):
    global number
    plt.figure()
    ax = df.plot(x='time', y='gyro_X',
                 title='Gyro',
                 colormap='Accent',
                 linewidth=0.6,
                 figsize=(20, 6)
                 )
    df.plot(x='time', y='gyro_Y', ax=ax,
            linewidth=0.6)
    df.plot(x='time', y='gyro_Z', ax=ax,
            linewidth=0.6)
    ax.grid(True)
    ax.set_xlabel('time[s]')
    ax.set_ylabel('[deg/s]')

    plt.savefig('fig/'+ str(number) + '.png')
    plt.close('all')
    number += 1

def vision_csv():
    # df = pd.read_csv('log_files/value_list1.csv', names=setup_variable.analysis_columns)
    # df2 = pd.read_csv('log_files/value_list2.csv', names=setup_variable.analysis_columns)

    log_list = glob.glob('log_files/value_list*.csv')
    for file_name in log_list:
        df = pd.read_csv(file_name, names=setup_variable.analysis_columns)
        start = df['timeStamp'][0]
        df['time'] = df['timeStamp'] - start
        set_plt_acc(df)
        set_plt_gyro(df)


# ================================= メイン関数　実行 ================================ #
if __name__ == '__main__':
    vision_csv()
