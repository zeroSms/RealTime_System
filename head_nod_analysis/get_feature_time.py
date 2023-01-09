#
# 特徴量計算スレッド
#
import numpy as np
import pandas as pd
from scipy.fft import rfft, fft

from . import setup_variable


def Sensor_name(sensor_name):
    if sensor_name == 'acc':
        return setup_variable.acc_columns
    elif sensor_name == 'gyro':
        return setup_variable.gyro_columns
    else:
        return setup_variable.axis_columns


feature_columns = []
def feature_name(sensor_name):
    axis_columns = Sensor_name(sensor_name)
    # 平均
    feature_columns.extend([('mean_' + name) for name in axis_columns])
    # 最大
    feature_columns.extend([('max_' + name) for name in axis_columns])
    # 最小
    feature_columns.extend([('min_' + name) for name in axis_columns])
    # 分散
    feature_columns.extend([('var_' + name) for name in axis_columns])
    # 中央値
    feature_columns.extend([('median_' + name) for name in axis_columns])
    # 第一四分位
    feature_columns.extend([('per25_' + name) for name in axis_columns])
    # 第三四分位
    feature_columns.extend([('per75_' + name) for name in axis_columns])
    # 四分位範囲
    feature_columns.extend([('per_range_' + name) for name in axis_columns])
    # 二乗平均平方根
    feature_columns.extend([('RMS_' + name) for name in axis_columns])

    # 相関係数
    if sensor_name == 'acc':
        coef_axis = ['acc_xy', 'acc_xz', 'acc_yz']
    elif sensor_name == 'gyro':
        coef_axis = ['gyro_xy', 'gyro_xz', 'gyro_yz']
    else:
        coef_axis = ['acc_xy', 'acc_xz', 'acc_yz', 'gyro_xy', 'gyro_xz', 'gyro_yz']
    feature_columns.extend(
        [('coef_' + name) for name in coef_axis])


def get_feature(window, sensor_name):
    axis_columns = Sensor_name(sensor_name)
    feature_list_mini = []
    df = pd.DataFrame(window, columns=setup_variable.analysis_columns)
    df = df.drop(['window_ID', 'timeStamp'], axis=1)
    df = df.astype('float')

    if sensor_name == 'acc':
        df = df.drop(setup_variable.gyro_columns, axis=1)
    elif sensor_name == 'gyro':
        df = df.drop(setup_variable.acc_columns, axis=1)

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
    feature_list_mini.extend(per_75 - per_25)
    # 二乗平均平方根
    square = np.square(df.values)
    feature_list_mini.extend(np.sqrt(np.mean(square, axis=0)))

    if sensor_name == 'all':
        # 相関係数(加速度)
        coef = df.iloc[:, 0:3].corr().values
        feature_list_mini.extend([coef[0, 1], coef[0, 2], coef[1, 2]])
        # 相関係数(角速度)
        coef = df.iloc[:, 3:6].corr().values
        feature_list_mini.extend([coef[0, 1], coef[0, 2], coef[1, 2]])
    else:
        # 相関係数
        coef = df.iloc[:, 0:3].corr().values
        feature_list_mini.extend([coef[0, 1], coef[0, 2], coef[1, 2]])
    # CrossCorrelation

    return feature_list_mini
