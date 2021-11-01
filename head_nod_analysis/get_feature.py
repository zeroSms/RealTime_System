#
# 特徴量計算スレッド
#
import math

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

    # Power Band
    for axis in axis_columns:
        feature_columns.extend(
            [(name + axis) for name in ['PB0-5', 'PB5-10', 'PB10-15', 'PB15-20', 'PB20-25', 'PB25-30',
                                        'PB30-35', 'PB35-40', 'PB40-45', 'PB45-50']])
    # Freq_Ent(周波数領域エントロピー)
    feature_columns.extend([('FE_' + name) for name in axis_columns])
    # Energy
    feature_columns.extend([('En_' + name) for name in axis_columns])


class Calc_FFT:
    # signal: 時系列信号（1軸分のみ）
    # sf: サンプリング周波数
    def __init__(self, signal):
        # パワースペクトル算出
        self.signal = signal.values.tolist()
        self.L = len(self.signal)  # 信号長
        self.freq = np.linspace(0, setup_variable.T, self.L)  # 周波数
        yf = np.abs(rfft(self.signal))  # 振幅スペクトル
        self.yf = yf * yf  # パワースペクトル

    # ============================ パワーバンドを求める関数 ============================== #
    # min_f: パワーバンドを算出する周波数の最小値
    # max_f: パワーバンドを算出する周波数の最小値
    def Calc_PowerBand(self, min_f, max_f):
        # 指定された周波数の探索
        n1, n2 = [], []
        for i in range(1, int(self.L / 2)):  # <- 直流成分は探索範囲から除外 & ナイキスト周波数まで
            n1.append(np.abs(self.freq[i] - min_f))
            n2.append(np.abs(self.freq[i] - max_f))
        min_id = np.argmin(n1)  # min_fと最も近い周波数が格納された配列番号取得
        max_id = np.argmin(n2)  # max_fと最も近い周波数が格納された配列番号取得

        # 指定された周波数のパワーバンド算出
        PB = np.sum(self.yf[min_id:max_id])

        return PB

    # ============================ 周波数領域エントロピーを求める関数 ============================== #
    def Calc_FreqEnt(self):
        # 手続き1
        a = 0
        for i in range(1, int(self.L / 2)):  # <- 直流成分を抜く & ナイキスト周波数まで
            a = a + self.yf[i]

        # 手続き2 & 3
        E = 0
        for i in range(1, int(self.L / 2)):  # <- 直流成分を抜く & ナイキスト周波数まで
            if self.yf[i] != 0:
                E = E + (self.yf[i] / a) * np.log2(self.yf[i] / a)
        E = -E
        return E

    # ============================ エネルギーを求める関数 ============================== #
    def Calc_Energy(self):
        # エネルギー算出
        En = np.mean(self.yf[1:int(self.L / 2)])  # 直流成分を除く & ナイキスト周波数まで

        return En

    # ============================ DC成分を求める関数 ============================== #
    def Calc_DC(self):
        DC = self.yf[0]  # 直流成分を除く & ナイキスト周波数まで

        return DC


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

    if sensor_name == 'acc' or 'gyro':
        # 相関係数
        coef = df.iloc[:, 0:3].corr().values
        feature_list_mini.extend([coef[0, 1], coef[0, 2], coef[1, 2]])
    else:
        # 相関係数(加速度)
        coef = df.iloc[:, 0:3].corr().values
        feature_list_mini.extend([coef[0, 1], coef[0, 2], coef[1, 2]])
        # 相関係数(角速度)
        coef = df.iloc[:, 3:6].corr().values
        feature_list_mini.extend([coef[0, 1], coef[0, 2], coef[1, 2]])
    # CrossCorrelation

    # 周波数領域
    axis_fft = {}
    for axis in axis_columns:
        axis_fft[axis] = Calc_FFT(df[axis])

    # Power Band
    FFT_PB = []
    for axis in axis_columns:
        # 5Hz刻みでパワーバンドを計算
        for i in range(0, 10):
            PB = axis_fft[axis].Calc_PowerBand(i * 5, (i + 1) * 5)
            FFT_PB.append(PB)
    feature_list_mini.extend(FFT_PB)

    # Freq Ent(周波数領域エントロピー)
    FFT_FE = []
    for axis in axis_columns:
        FE = axis_fft[axis].Calc_FreqEnt()
        FFT_FE.append(FE)
    feature_list_mini.extend(FFT_FE)

    # Energy
    FFT_En = []
    for axis in axis_columns:
        En = axis_fft[axis].Calc_Energy()
        FFT_En.append(En)
    feature_list_mini.extend(FFT_En)

    return feature_list_mini
