#
# 特徴量計算スレッド
#
import numpy as np
import pandas as pd
from . import setup_variable

feature_columns = []
# 平均
feature_columns.extend([('mean_' + name) for name in setup_variable.axis_columns])
# 最大
feature_columns.extend([('max_' + name) for name in setup_variable.axis_columns])
# 最小
feature_columns.extend([('min_' + name) for name in setup_variable.axis_columns])
# 分散
feature_columns.extend([('var_' + name) for name in setup_variable.axis_columns])
# 中央値
feature_columns.extend([('median_' + name) for name in setup_variable.axis_columns])
# 第一四分位
feature_columns.extend([('per25_' + name) for name in setup_variable.axis_columns])
# 第三四分位
feature_columns.extend([('per75_' + name) for name in setup_variable.axis_columns])
# 四分位範囲
feature_columns.extend([('per_range_' + name) for name in setup_variable.axis_columns])
# 二乗平均平方根
feature_columns.extend([('RMS_' + name) for name in setup_variable.axis_columns])


def get_feature(window):
    feature_list_mini = []
    df = pd.DataFrame(window, columns=setup_variable.analysis_columns)
    df = df.drop(['window_ID', 'timeStamp'], axis=1)
    df = df.astype('float')

    # 平均
    feature_list_mini.extend(np.mean(df.values, axis=0))

    # 最大
    feature_list_mini.extend(np.max(df.values, axis=0))

    # 最小
    feature_list_mini.extend(np.min(df.values, axis=0))

    # 分散
    feature_list_mini.extend(np.var(df.values, axis=0))

    # 中央値
    feature_list_mini.extend(np.median(df.values, axis=0))

    # 第一四分位
    per_25 = np.percentile(df.values, 25, axis=0)
    feature_list_mini.extend(per_25)

    # 第三四分位
    per_75 = np.percentile(df.values, 75, axis=0)
    feature_list_mini.extend(per_75)

    # 四分位範囲
    feature_list_mini.extend(per_75-per_25)

    # 二乗平均平方根
    square = np.square(df.values)
    feature_list_mini.extend(np.sqrt(np.mean(square, axis=0)))

    return feature_list_mini
