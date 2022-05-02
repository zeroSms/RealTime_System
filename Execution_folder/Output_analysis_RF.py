#
# データ処理スレッド
#
import csv
import glob
import os
import shutil
import time

import pandas as pd

# 自作ライブラリ
from head_nod_analysis import setup_variable, feature_selection

# 分類モデル
from sklearn.ensemble import RandomForestClassifier  # ランダムフォレスト分類クラス
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict, GridSearchCV  # 層化K分割用クラス
from sklearn.metrics import classification_report

# 図の描画
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid', {'linestyle.grid': '--'})

# ================================= パスの取得 ================================ #
path = setup_variable.path

# ============================ ウィンドウ処理スレッド ============================== #
# ウィンドウ単位の処理用定数
sensor_name = 'all'


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
    ex_num = input('実験番号：')
    data_set = 'main'

    # Resultの初期化
    make_file = path + '\\Result\\feature' + str(ex_num)
    if os.path.exists(make_file):
        shutil.rmtree(make_file)
    os.makedirs(make_file)

    # 特徴量選択
    analysis_data_file = path + '\\data_set\\analysis_files\\feature_selection\\' + sensor_name + '\\None'

    # 特徴量/正解データ取得
    X, y = feature_download()
    X_value = X.values

    # ハイパーパラメータ
    param_grid = {'max_depth': [i for i in range(10, 101, 10)],
                  'n_estimators': [i for i in range(10, 101, 10)]}

    # 層化k分割交差検証
    FOLD = setup_variable.FOLD  # 交差検証分割数
    random_state = setup_variable.random_state
    stratifiedkfold = StratifiedKFold(n_splits=FOLD, shuffle=True, random_state=random_state)

    # 最適なパラメータの探索
    score_list = []
    predict_time_list = []
    roop = 0
    for max_depth in param_grid['max_depth']:
        score_list.append([])
        predict_time_list.append([])

        for n_estimators in param_grid['n_estimators']:

            # 分類モデルの適用
            clf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=random_state)

            # 層化k分割交差検証(実行)
            start = time.time()
            y_pred = cross_val_predict(clf, X_value, y, cv=stratifiedkfold)
            elapsed_time = time.time() - start

            # 予測結果リストの出力
            test_score = classification_report(y, y_pred, target_names=['others', 'nod', 'shake'], digits=3, output_dict=True)

            score_list[-1].append(test_score['macro avg']['f1-score'])
            predict_time_list[-1].append(elapsed_time)

            roop += 1
            print(roop)

    # ヒートマップ
    plt.figure()
    df = pd.DataFrame(data=score_list, index=param_grid['max_depth'], columns=param_grid['n_estimators'])
    sns.heatmap(df, cmap='Blues', annot=True, fmt='.3g', square=True)
    plt.xlabel('n_estimators')
    plt.ylabel('max_depth')
    plt.savefig(make_file+'\\heatmap_f1-score')

    plt.figure()
    df = pd.DataFrame(data=predict_time_list, index=param_grid['max_depth'], columns=param_grid['n_estimators'])
    sns.heatmap(df, cmap='Blues', annot=True, fmt='.3g', square=True)
    plt.xlabel('n_estimators')
    plt.ylabel('max_depth')
    plt.savefig(make_file+'\\heatmap_predict_time')

    # パラメータの出力
    paramater = {'data_set': data_set,
                 'サンプリング周波数': setup_variable.T,
                 'オーバーラップ率': setup_variable.OVERLAP,
                 'ウィンドウサイズ': setup_variable.N,
                 'ウィンドウラベル閾値': setup_variable.threshold,
                 'random_state': setup_variable.random_state}
    with open(make_file + '\\paramater' + str(ex_num) + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for k, v in paramater.items():
            writer.writerow([k, v])

    with open(make_file + '\\predict_time' + str(ex_num) + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for k, v in paramater.items():
            writer.writerow([k, v])
