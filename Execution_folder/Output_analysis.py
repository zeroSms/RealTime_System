#
# データ処理スレッド
#
import csv
import glob
import os
import pickle
import shutil
import time
from _csv import reader
import numpy as np
import pandas as pd

# 自作ライブラリ
from head_nod_analysis import setup_variable, get_feature, feature_selection, process_data

# 分類モデル
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report

# 図の描画
import matplotlib.pyplot as plt
from head_nod_analysis import view_Confusion_matrix
import seaborn as sns

sns.set_style('whitegrid', {'linestyle.grid': '--'})

analysis_csv = [setup_variable.analysis_columns]  # windowデータの追加
answer_list = []  # 正解データリスト（windowごと）
feature_list = []  # 特徴量抽出データ

# ================================= パスの取得 ================================ #
path = setup_variable.path


# ================================= 分析ファイルの出力 ================================ #
def output_files(X):
    # Resultの初期化
    analysis_data_file = path + '\\data_set\\analysis_files\\' + str(ex_num)
    if os.path.exists(analysis_data_file):
        shutil.rmtree(analysis_data_file)
    os.makedirs(analysis_data_file)

    # ウィンドウ処理したデータの出力
    to_csv_name = analysis_data_file + '\\window_list.csv'
    df_analysis.to_csv(to_csv_name)

    # 正解データの出力
    with open(analysis_data_file + '\\answer_list.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(answer_list)

    # 特徴量データの出力(選択なし)
    with open(analysis_data_file + '\\feature_list.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(feature_list)

    # 特徴量データの出力(選択あり)
    to_csv_name = analysis_data_file + '\\feature_list_selection.csv'
    X.to_csv(to_csv_name)

    # 学習モデルの保存
    pickle.dump(clf, open(analysis_data_file + '\\trained_model.pkl', 'wb'))


# ============================ ウィンドウ処理スレッド ============================== #
# ウィンドウ単位の処理用定数
sensor_name = 'all'
data_set = '100Hz'
T = setup_variable.T  # サンプリング周期 [Hz]
N = setup_variable.N  # ウィンドウサイズ
OVERLAP = setup_variable.OVERLAP  # オーバーラップ率 [%]
window_data = []
get_window = []
window_num = 0  # window番号


# ウィンドウ処理を行う
def process_window(data_queue):
    global window_data, window_num, get_window
    not_dup = int(N * (1 - OVERLAP / 100))  # 重複しない部分の個数
    if not_dup < 1:
        not_dup = 1

    # サンプル数（N）分のデータを格納するリスト（window_data）の作成
    for _ in range(not_dup):
        # 重複しない部分のデータはキューから削除
        window_data.append(data_queue.pop(0))
    for i in range(N - not_dup):
        window_data.append(data_queue[i])

    if window_data:
        get_window = window_data
        window_data = []  # ウィンドウをリセット
        window_num += 1
        return get_window


# 各ファイルごとにウィンドウ処理を実行，結果をCSV出力
def do_process_window():
    # data_filesの初期化
    rm_file = path + '\\data_set\\analysis_files\\data_files'
    shutil.rmtree(rm_file)
    os.makedirs(rm_file)
    # logファイルのコピー
    glob_file = path + '\\data_set\\log_files\\' + data_set + '\\value_list*.csv'  # 全ファイル
    log_list = glob.glob(glob_file)
    for file_name in log_list:
        with open(file_name, 'r') as f:
            csv_reader = reader(f)
            log_data = list(csv_reader)
        with open(path + '\\data_set\\analysis_files\\data_files\\' + file_name.split('\\')[-1], 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerows(log_data)

    # CSVファイル出力
    glob_file = path + '\\data_set\\analysis_files\\data_files\\value_list*.csv'
    file_list = glob.glob(glob_file)
    for file_name in file_list:
        df_data_queue = pd.read_csv(file_name, names=(
            setup_variable.analysis_columns))

        # キュー内のデータ数がサンプル数を超えている間作動
        data_queue = df_data_queue.values.tolist()
        while len(data_queue) > N:
            window = process_window(data_queue)
            if window:
                # 特徴量抽出
                feature_list.append(get_feature.get_feature(window, sensor_name))
                # ウィンドウラベルの付与，正解ラベルデータの作成
                result_window, answer_num = process_data.label_shape(window)
                answer_list.append(answer_num)
                analysis_csv.extend(result_window)


# ================================= メイン関数　実行 ================================ #
FOLD = setup_variable.FOLD  # 交差検証分割数
importance_ave = []
sum_test = []
sum_pred = []
score_dict = {}
PCA_bar = []

if __name__ == '__main__':
    ex_num = input('実験番号：')
    feature_check = input('特徴量選択[1/2/n]：')

    # Resultの初期化
    make_file = path + '\\Result\\feature' + str(ex_num)
    if os.path.exists(make_file):
        shutil.rmtree(make_file)
    os.makedirs(make_file)

    # 特徴量リスト
    get_feature.feature_name(sensor_name)
    # ウィンドウ処理
    do_process_window()

    # TimeStampラベルを削除
    df_analysis = pd.DataFrame(analysis_csv[1:], columns=analysis_csv[0])
    df_analysis = df_analysis.drop('timeStamp', axis=1)

    # 正解データ取得
    X = pd.DataFrame(feature_list, columns=get_feature.feature_columns)
    y = pd.Series(data=np.array(answer_list))

    # 分類モデルの適用
    max_depth = setup_variable.max_depth
    n_estimators = setup_variable.n_estimators
    random_state = setup_variable.random_state
    clf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=random_state)

    # 特徴量選択
    if feature_check == '1':
        method = feature_selection.Wrapper_Method(clf, X, y, make_file, ex_num)
        X = method.RFE_CV()
    elif feature_check == '2':
        method = feature_selection.Embedded_Method(clf, X, y, make_file, ex_num)
        X = method.SFM()
    X_value = X.values

    # 分析ファイルの出力
    clf.fit(X_value, y)
    output_files(X)

    # 層化k分割交差検証
    start = time.time()
    skf = StratifiedKFold(n_splits=FOLD)
    for train_id, test_id in skf.split(X_value, y):
        train_x, test_x = X_value[train_id], X_value[test_id]
        train_y, test_y = y[train_id], y[test_id]

        clf.fit(train_x, train_y)

        # テストセットの精度検証
        pred_test = clf.predict(test_x)
        test_score = classification_report(test_y, pred_test, target_names=['others', 'nod', 'shake'], output_dict=True)
        # print(classification_report(test_y, pred_test, target_names=['others', 'nod', 'shake']))
        for k1 in test_score.keys():
            if not (k1 in score_dict):
                if k1 == 'accuracy':
                    score_dict[k1] = 0
                else:
                    score_dict[k1] = {}
            if k1 == 'accuracy':
                score_dict[k1] += test_score[k1]
            else:
                for k2 in test_score[k1].keys():
                    if not (k2 in score_dict[k1]):  score_dict[k1][k2] = 0
                    score_dict[k1][k2] += float(test_score[k1].get(k2, 0))

        importance_ave.append(clf.feature_importances_.tolist())

        sum_test.extend(test_y)
        sum_pred.extend(pred_test)

    elapsed_time = time.time() - start

    # 混同行列
    view_Confusion_matrix.print_cmx(sum_test, sum_pred, make_file, ex_num)

    # 交差検証結果（平均）
    for k1 in score_dict.keys():
        if k1 == 'accuracy':
            score_dict[k1] /= FOLD
        else:
            for k2 in score_dict[k1].keys():
                if k2 != 'support':
                    score_dict[k1][k2] /= FOLD
    df = pd.DataFrame(score_dict).T
    df = df.round(2)
    df = df.astype({'support': 'int'})
    df['elapsed_time'] = elapsed_time
    df.to_csv(make_file + '\\result_score' + str(ex_num) + '.csv')
    print(df)

    # ランダムフォレストの説明変数の重要度をデータフレーム化
    fea_rf_imp = pd.DataFrame({'imp': np.mean(np.array(importance_ave), axis=0), 'col': X.columns})
    fea_rf_imp = fea_rf_imp.sort_values(by='imp', ascending=False)
    fea_rf_imp.to_csv(make_file + '\\features_importance' + str(ex_num) + '.csv')

    # ランダムフォレストの重要度を可視化
    fig = plt.figure(figsize=(10, 7))
    sns.barplot(x='imp', y='col', data=fea_rf_imp, orient='h')
    plt.title('Random Forest - Feature Importance', fontsize=28)
    plt.ylabel('Features', fontsize=18)
    plt.xlabel('Importance', fontsize=18)
    fig.savefig(make_file + '\\features_importance' + str(ex_num) + '.png')

    # パラメータの出力
    paramater = {'data_set': data_set,
                 'サンプリング周波数': setup_variable.T,
                 'オーバーラップ率': setup_variable.OVERLAP,
                 'ウィンドウサイズ': setup_variable.N,
                 'ウィンドウラベル閾値': setup_variable.threshold,
                 'max_depth': setup_variable.max_depth,
                 'n_estimator': setup_variable.n_estimators,
                 'random_state': setup_variable.random_state}
    with open(make_file + '\\paramater' + str(ex_num) + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for k, v in paramater.items():
            writer.writerow([k, v])

