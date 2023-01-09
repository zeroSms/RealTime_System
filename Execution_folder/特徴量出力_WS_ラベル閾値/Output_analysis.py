#
# データ処理スレッド
#
import csv
import glob
import os
import pickle
import shutil

# 自作ライブラリ
from head_nod_analysis import setup_variable, feature_selection

# 分類モデル
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # ランダムフォレスト分類クラス
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict  # 層化K分割用クラス
from sklearn.metrics import classification_report

from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline

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

# ============================ ウィンドウ処理スレッド ============================== #
def feature_download():
    # 特徴量リストの出力
    glob_file = analysis_data_file + '\\feature_list_selection*.csv'  # 全ファイル
    log_list = glob.glob(glob_file)
    df = []
    for file_name in log_list:
        df.append(pd.read_csv(file_name, header=0, index_col=0))
    X = pd.concat(df)

    # 正解リストの出力
    answer_file = analysis_data_file + '\\answer_list.csv'  # 全ファイル
    with open(answer_file) as f:
        reader = csv.reader(f)
        for row in reader:
            y = row

    return X, y


# ================================= メイン関数　実行 ================================ #
if __name__ == '__main__':
    ex_num = int(input('実験番号：'))
    sensor_name = input('データセット[all/acc/gyro]：')
    T = setup_variable.T                                # サンプリング周波数
    OVERLAP = setup_variable.OVERLAP                    # オーバーラップ率

    N_list = [128, 64, 32, 16, 8]
    threshold_list = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    macro_avg_dict = {}

    for N in N_list:
        macro_avg_dict[N] = []
        for threshold in threshold_list:
            print('\n=====================================================')
            print('WS:{} threshold:{}'.format(N, threshold))
            # ファイル名の設定
            feature_file_name = 'WS' + str(N) + '_threshold' + str(threshold)

            # Resultの初期化
            make_file = path + '\\Result\\feature_' + str(ex_num)
            if os.path.exists(make_file):
                shutil.rmtree(make_file)
            os.makedirs(make_file)
            ex_num += 1     # ex_numに1を加算

            # 分類モデルの適用
            random_state = setup_variable.random_state
            clf = RandomForestClassifier(random_state=random_state)

            analysis_data_file = path + '\\data_set\\analysis_files\\feature_selection\\' + sensor_name + '\\' + feature_file_name

            # 特徴量/正解データ取得
            X, y = feature_download()
            X_value = X.values
            classifer = Pipeline([('estimator', clf)])

            # 分析ファイルの出力（全データ）　⇒　リアルタイム分析用
            classifer.fit(X_value, y)
            trained_file = path + '\\data_set\\analysis_files\\trained_model'
            pickle.dump(classifer['estimator'], open(trained_file + '\\trained_model' + str(ex_num) + '.pkl', 'wb'))

            try:

                # 層化k分割交差検証
                FOLD = setup_variable.FOLD  # 交差検証分割数
                stratifiedkfold = StratifiedKFold(n_splits=FOLD, shuffle=True, random_state=random_state)

                # 層化k分割交差検証(予測結果リストの出力)
                y_pred = cross_val_predict(classifer, X_value, y, cv=stratifiedkfold)
                test_score = classification_report(y, y_pred, target_names=['others', 'nod', 'shake'], digits=3, output_dict=True)
                macro_avg_dict[N].append(round(test_score['macro avg']['f1-score'], 3))
                print(test_score)

                # 混同行列
                view_Confusion_matrix.print_cmx(y, y_pred, make_file, ex_num)

                # 交差検証結果（平均）
                df = pd.DataFrame(test_score).T
                df = df.round(3)
                df = df.astype({'support': 'int'})
                # df['sum_fit_time'] = sum(scores['fit_time'])
                df.to_csv(make_file + '\\result_score' + str(ex_num) + '.csv')
                print(df)

                # パラメータの出力
                paramater = {'data_set': sensor_name,
                             'サンプリング周波数': T,
                             'オーバーラップ率': OVERLAP,
                             'ウィンドウサイズ': N,
                             'ウィンドウラベル閾値': threshold,
                             'random_state': setup_variable.random_state}
                with open(make_file + '\\paramater' + str(ex_num) + '.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    for k, v in paramater.items():
                        writer.writerow([k, v])
            except ValueError as e:
                print('catch ValueError:', e)
                macro_avg_dict[N].append(0)

    # ヒートマップ出力
    print(macro_avg_dict)
    macro_avg_list = [macro_avg for macro_avg in macro_avg_dict.values()]

    print(macro_avg_list)
    df = pd.DataFrame(data=macro_avg_list, index=N_list, columns=threshold_list)
    print(df)
    #    A  B  C  D
    # a -8 -7 -6 -5
    # b -4 -3 -2 -1
    # c  0  1  2  3
    # d  4  5  6  7

    plt.figure()
    sns.heatmap(df, annot=True, square=True, cmap='Blues', fmt='.3f', vmax=1)
    plt.savefig(path + '\\Execution_folder\\特徴量出力_WS_ラベル閾値\\seaborn_heatmap_dataframe.png')
    
    