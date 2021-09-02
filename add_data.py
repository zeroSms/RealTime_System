#
# データ取得スレッド
#

import csv
import asyncio
import platform
from bleak import discover
import struct
from bleak import BleakClient
import numpy as np
import time
import stop

# ウィンドウ単位の処理用定数
N = 20              # サンプル数
OVERLAP = 50        # オーバーラップ率 [%]
T = 10              # サンプリング周期 [Hz]
byte_sample = 0

sec = 0             # 秒数
data_queue = []     # 保存用変数

# async def run():
#     devices = await discover()
#     for d in devices:
#         print(d)
#
# loop = asyncio.get_event_loop()
# loop.run_until_complete(run())

# eSENSE キャラクタリスティックUUID
UUID7 = "0000ff07-0000-1000-8000-00805f9b34fb"  # サンプリング開始/終了 (R/W)
UUID8 = "0000ff08-0000-1000-8000-00805f9b34fb"  # データ取得 (R)
UUIDE = "0000ff0e-0000-1000-8000-00805f9b34fb"  # フルスケールレンジ・ローパスフィルタの取得/変更 (R/W)

# def sampling_byte(T):
#     if T = 10:
#         byte_sample = bytearray([0x53, 0x0d, 0x02, 0x01, 0x0a])

# センサの設定を行うクラス
class Sensor:
    def __init__(self, address, loop):
        self.address = address
        self.loop = loop
        self.window = []

    # Notify 呼び出し関数
    def callback(sender, value):
        TimeStamp = time.time()
        shape_int16 = struct.unpack('>bbbbhhhhhh', value)

        value_acc_X, value_acc_Y, value_acc_Z = shape_int16[7], shape_int16[8], shape_int16[9]
        value_gyro_X, value_gyro_Y, value_gyro_Z = shape_int16[4], shape_int16[5], shape_int16[6]
        # データ保存
        data_queue.append([TimeStamp, value_acc_X, value_acc_Y, value_acc_Z, value_gyro_X, value_gyro_Y, value_gyro_Z])
        # 表示
        print("Acc: {0} {1} {2}".format(value_acc_X, value_acc_Y, value_acc_Z))
        print("Gyro: {0} {1} {2}".format(value_gyro_X, value_gyro_Y, value_gyro_Z))

    # センサからデータを取得
    async def ReadSensor(self):
        async with BleakClient(self.address, loop=self.loop) as client:
            x = await client.is_connected()
            print("Connected: {0}".format(x))

            # サンプリング開始 20Hz
            await client.write_gatt_char(UUID7, bytearray([0x53, 0x17, 0x02, 0x01, 0x14]), response=True)

            await client.start_notify(UUID8, Sensor.callback)

            # 5秒後に終了
            # await asyncio.sleep(1.0, loop=self.loop)
            while stop.Stop():
                await asyncio.sleep(1.0, loop=self.loop)
            await client.stop_notify(UUID8)
            # サンプリング終了
            await client.write_gatt_char(UUID7, bytearray([0x53, 0x02, 0x02, 0x00, 0x00]), response=True)
            
    # ウィンドウ処理を行う
    def process_window(self):
        while stop.stop_flg:
            # キュー内のデータ数がサンプル数を超えたら作動
            if len(data_queue) > N:
                notdup = int(N * (1 - OVERLAP / 100))  # 重複しない部分の個数
                if notdup < 1:
                    notdup = 1  # NotDupが0だと初期値の無限ループになる

                # サンプル数（N）分のデータを格納するリスト（window）の作成
                for _ in range(notdup):
                    # 重複しない部分のデータはキューから削除
                    self.window.append(data_queue.pop(0))
                for i in range(N - notdup):
                    self.window.append(data_queue[i])

                if self.window != []:
                    w = self.window
                    self.window = []    # ウィンドウをリセット
                    return w

def AddData():
    filename = "data_files/others/value_list" + input("file_number 入力：") + ".csv"
    print(filename)
    address = (
        # discovery.pyでみつけたtoio Core Cubeのデバイスアドレスをここにセットする
        "00:04:79:00:0D:00"  # Windows か Linux のときは16進6バイトのデバイスアドレスを指定
        # if platform.system() != "Darwin"
        # else "243E23AE-4A99-406C-B317-18F1BD7B4CBE"  # macOSのときはmacOSのつけるUUID
    )
    loop = asyncio.get_event_loop()
    loop.run_until_complete(Sensor(address, loop).ReadSensor())

    with open(filename, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(data_queue)


if __name__ == '__main__':
    AddData()
