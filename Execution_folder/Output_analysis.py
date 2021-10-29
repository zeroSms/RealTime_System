#
# データ処理スレッド
#
import csv
import glob
import os
import shutil
import time
from _csv import reader
import numpy as np
from collections import Counter
import pandas as pd

# 自作ライブラリ
from head_nod_analysis import setup_variable, get_feature

# 分類モデル
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

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


# ============================ データ整形スレッド ============================== #
def label_shape(window):
    window_T = np.array(window).T  # 転置　（labelごとの個数を計算するため）
    label_num = Counter(window_T[0])  # labelごとの個数を計算

    # 正解データファイルの出力
    if label_num['nod'] > int(len(window) * 0.3):
        answer_list.append(1)
    elif label_num['shake'] > int(len(window) * 0.3):
        answer_list.append(2)
    else:
        answer_list.append(0)
    window_T[0] = [window_num] * len(window)  # window_IDの追加

    return window_T.T


# ============================ ウィンドウ処理スレッド ============================== #
# ウィンドウ単位の処理用定数
T = setup_variable.T  # サンプリング周期 [Hz]
N = setup_variable.N  # ウィンドウサイズ(1.28秒)
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
    # glob_file = path + '\\data_set\\log_files\\100Hz\\value_list*.csv'  # 全ファイル
    glob_file = path + '\\data_set\\log_files\\main\\value_list*.csv'  # 全ファイル
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
                # feature_list.append(get_feature.get_feature(window))
                feature_list.append(get_feature.get_feature(window))
                analysis_csv.extend(label_shape(window))


# =========================================== メイン関数　実行 ============================================== #
FOLD = setup_variable.FOLD  # 交差検証分割数
importance_ave = []
sum_test = []
sum_pred = []
score_dict = {}
PCA_bar = []

if __name__ == '__main__':
    ex_num = input('実験番号：')
    feature_check = input('特徴量選択[y/n]：')

    # Resultの初期化
    make_file = path + '\\Result\\feature' + str(ex_num)
    if os.path.exists(make_file):
        shutil.rmtree(make_file)
    os.makedirs(make_file)

    # ウィンドウ処理
    do_process_window()

    # TimeStampラベルを削除
    df_analysis = pd.DataFrame(analysis_csv[1:], columns=analysis_csv[0])
    df_analysis = df_analysis.drop('timeStamp', axis=1)

    # ウィンドウ処理したデータの出力
    to_csv_name = path + '\\data_set\\analysis_files\\window_files\\window_list' + ex_num + '.csv'
    df_analysis.to_csv(to_csv_name)

    # 正解データの出力
    with open(path + '\\data_set\\analysis_files\\answer_files\\answer_list' + ex_num + '.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(answer_list)

    # 特徴量データの出力
    with open(path + '\\data_set\\analysis_files\\feature_files\\feature_list' + ex_num + '.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(feature_list)

    # 正解データ取得
    X = pd.DataFrame(feature_list, columns=get_feature.feature_columns)
    y = pd.Series(data=np.array(answer_list))

    clf = RandomForestClassifier(max_depth=30, n_estimators=30, random_state=42)

    if feature_check == 'y':
        # 特徴量削減
        min_features_select = 10
        selector = RFECV(clf, min_features_to_select=min_features_select, cv=10)
        # selector = RFE(clf, n_features_to_select=min_features_select)
        X_new = pd.DataFrame(selector.fit_transform(X, y),
                             columns=X.columns.values[selector.get_support()])
        print(len(X.columns.values[selector.get_support()]))
        result = pd.DataFrame(selector.get_support(), index=X.columns.values, columns=['False: dropped'])
        result['ranking'] = selector.ranking_
        result.to_csv(make_file + '\\feature_rank' + str(ex_num) + '.csv')

        # Plot number of features VS. cross-validation scores
        fig = plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(min_features_select,
                       len(selector.grid_scores_) + min_features_select),
                 selector.grid_scores_)
        fig.savefig(make_file + '\\features_score' + str(ex_num) + '.png')
        X = X_new

    X_value = X.values

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

    end = time.time()

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
    df.to_csv(make_file + '\\result_score' + str(ex_num) + '.csv')
    print(df)

    # ランダムフォレストの説明変数の重要度をデータフレーム化
    fea_rf_imp = pd.DataFrame({'imp': np.mean(np.array(importance_ave), axis=0), 'col': X.columns})
    fea_rf_imp = fea_rf_imp.sort_values(by='imp', ascending=False)

    # ランダムフォレストの重要度を可視化
    fig = plt.figure(figsize=(10, 7))
    sns.barplot(x='imp', y='col', data=fea_rf_imp, orient='h')
    plt.title('Random Forest - Feature Importance', fontsize=28)
    plt.ylabel('Features', fontsize=18)
    plt.xlabel('Importance', fontsize=18)
    fig.savefig(make_file + '\\features_importance' + str(ex_num) + '.png')
    plt.show()

    print("elapsed_time:{0}".format(end-start) + "[sec]")

