#
# データ取得スレッド
#

import asyncio
from bleak import discover, BleakClient
import struct
import time

# 自作ライブラリ
from . import enter_label, setup_variable, stop

# ============================ 変数宣言部 ============================== #
# ウィンドウ単位の処理用定数
T = setup_variable.T  # サンプリング周期 [Hz]
N = setup_variable.N  # ウィンドウサイズ(0.64秒)
OVERLAP = setup_variable.OVERLAP  # オーバーラップ率 [%]
byte_sample = bytearray([0x53, 0x03, 0x02, 0x01, 0x00])  # UUID7　書き込み用バイト（サンプリング開始）
eSense_address = 0
w = []
data_queue = []  # 分析用データ格納
log_data = []  # 保存用データ格納
sensor = 0

# eSENSE キャラクタリスティックUUID
UUID7 = "0000ff07-0000-1000-8000-00805f9b34fb"  # サンプリング開始/終了 (R/W)
UUID8 = "0000ff08-0000-1000-8000-00805f9b34fb"  # データ取得 (R)
UUIDE = "0000ff0e-0000-1000-8000-00805f9b34fb"  # フルスケールレンジ・ローパスフィルタの取得/変更 (R/W)


# ============================ eSenseの処理 ============================== #
# UUID7　書き込み用バイト（サンプリング開始）の周波数加算
def sampling_byte():
    byte_sample[1] += T
    byte_sample[4] += T
    return byte_sample

# eSenseのアドレスを取得
async def search_eSense():
    global eSense_address
    eSense_flg = True
    while eSense_flg:
        devices = await discover()
        for d in devices:
            if 'eSense' in str(d):
                eSense_flg = False
                print(d)
                eSense_address = str(d).rsplit(':', 1)

# センサの設定を行うクラス
class Sensor:
    def __init__(self, address, loop):
        self.address = address
        self.loop = loop
        self.window = []

    # Notify 呼び出し関数
    def callback(sender, value):
        TimeStamp = time.time()
        shape_int16 = struct.unpack('>bbbbhhhhhh', value)  # Binary変換　デコード

        # accの値をm/s^2に変換
        value_acc_X = shape_int16[7] / 8192 * 9.80665
        value_acc_Y = shape_int16[8] / 8192 * 9.80665
        value_acc_Z = shape_int16[9] / 8192 * 9.80665

        # gyroの値をdeg/sに変換
        value_gyro_X = shape_int16[4] / 65.5
        value_gyro_Y = shape_int16[5] / 65.5
        value_gyro_Z = shape_int16[6] / 65.5

        # 合成軸を計算
        acc_xyz = (value_acc_X**2 + value_acc_Y**2 + value_acc_Z**2) ** 0.5

        # データ保存
        data_queue.append([enter_label.label_flg, TimeStamp,
                           value_acc_X, value_acc_Y, value_acc_Z, value_gyro_X, value_gyro_Y, value_gyro_Z, acc_xyz])
        log_data.append([enter_label.label_flg, TimeStamp,
                         value_acc_X, value_acc_Y, value_acc_Z, value_gyro_X, value_gyro_Y, value_gyro_Z, acc_xyz])
        # 表示
        # print("Acc: {0} {1} {2}".format(value_acc_X, value_acc_Y, value_acc_Z))
        # print("Gyro: {0} {1} {2}".format(value_gyro_X, value_gyro_Y, value_gyro_Z))

    # センサからデータを取得
    async def ReadSensor(self):
        async with BleakClient(self.address, loop=self.loop) as client:
            x = await client.is_connected()
            print("Connected: {0}".format(x))

            # サンプリング開始
            await client.write_gatt_char(UUID7, sampling_byte(), response=True)

            await client.start_notify(UUID8, Sensor.callback)

            # enterで終了
            while stop.stop_flg:
                await asyncio.sleep(1.0, loop=self.loop)
            await client.stop_notify(UUID8)
            # サンプリング終了
            await client.write_gatt_char(UUID7, bytearray([0x53, 0x02, 0x02, 0x00, 0x00]), response=True)

    # ウィンドウ処理を行う
    def process_window(self):
        global w
        while stop.stop_flg:
            # キュー内のデータ数がサンプル数を超えたら作動
            if len(data_queue) > N:
                not_dup = int(N * (1 - OVERLAP / 100))  # 重複しない部分の個数
                if not_dup < 1:
                    not_dup = 1

                # サンプル数（N）分のデータを格納するリスト（window）の作成
                for _ in range(not_dup):
                    # 重複しない部分のデータはキューから削除
                    self.window.append(data_queue.pop(0))
                for i in range(N - not_dup):
                    self.window.append(data_queue[i])

                if self.window:
                    w = self.window
                    self.window = []  # ウィンドウをリセット
                    return w


# ============================ データ取得スレッド ============================== #
def AddData(address, loop):
    global sensor
    sensor = Sensor(address, loop)
    asyncio.set_event_loop(loop)
    loop.run_until_complete(sensor.ReadSensor())

# def get_address():
#     loop = asyncio.get_event_loop()
#     loop.run_until_complete(search_eSense())

# def AddData1():
#     filename = "data_files/others/value_list" + input("file_number 入力：") + ".csv"
#     print("ファイル名：" + filename)
#
#     loop = asyncio.get_event_loop()
#     loop.run_until_complete(search_eSense())
#     address = eSense_address[0]
#     loop = asyncio.get_event_loop()
#     loop.run_until_complete(Sensor(address, loop).ReadSensor())
#
#     with open(filename, 'w') as f:
#         writer = csv.writer(f, lineterminator='\n')
#         writer.writerows(data_queue)
#
#
# if __name__ == '__main__':
#     AddData1()
