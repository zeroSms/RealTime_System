#
# 特徴量計算スレッド
#
import numpy as np
import pandas as pd
from scipy.fft import rfft, fft

from . import setup_variable

PB_list = ['PB0-5', 'PB5-10']

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

    # Power Band
    for axis in axis_columns:
        feature_columns.extend(
            [(name + axis) for name in PB_list])
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

    # 周波数領域
    axis_fft = {}
    for axis in axis_columns:
        axis_fft[axis] = Calc_FFT(df[axis])

    # Power Band
    FFT_PB = []
    for axis in axis_columns:
        # 5Hz刻みでパワーバンドを計算
        for i in range(0, len(PB_list)):
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
