
# 図の描画
from head_nod_analysis import view_Confusion_matrix
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ================================= パスの取得 ================================ #
# path = setup_variable.path
path = 'C:\\Users\\perun\\PycharmProjects\\RealTime_System'



# ================================= メイン関数　実行 ================================ #
if __name__ == '__main__':
    N_list = [128, 64, 32, 16, 8]
    threshold_list = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    # ヒートマップ出力
    macro_avg_list = [[0.000, 0.000, 0.377, 0.568, 0.722, 0.832, 0.844, 0.885, 0.931, 0.947],
                      [0.000, 0.692, 0.794, 0.859, 0.887, 0.904, 0.928, 0.942, 0.948, 0.952],
                      [0.000, 0.874, 0.907, 0.921, 0.930, 0.936, 0.942, 0.949, 0.951, 0.949],
                      [0.000, 0.513, 0.516, 0.517, 0.519, 0.520, 0.526, 0.526, 0.527, 0.527],
                      [0.000, 0.429, 0.430, 0.433, 0.437, 0.437, 0.438, 0.440, 0.440, 0.441]
                      ]
    df = pd.DataFrame(data=macro_avg_list, index=N_list, columns=threshold_list).T
    print(df)
    #    A  B  C  D
    # a -8 -7 -6 -5
    # b -4 -3 -2 -1
    # c  0  1  2  3
    # d  4  5  6  7

    plt.figure()
    sns.heatmap(df, annot=True, cmap='Blues', fmt='.3f', vmax=1, linewidths=0.5)

    # *以下2行がポイント*  X,Y軸ラベルを追加
    plt.xlabel("window size")
    plt.ylabel("threshold")
    # ヒートマップ保存
    plt.savefig(path + '\\Execution_folder\\特徴量出力_WS_ラベル閾値\\seaborn_heatmap_dataframe_2.png')


