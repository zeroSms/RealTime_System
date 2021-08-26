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

# ウィンドウ単位の処理用定数
N = 5                   # サンプル数
OVERLAP = 50            # オーバーラップ率 [%]
T = 0.1                 # サンプリング周期 [s]

# ログファイル用変数
log_list = []
sec = 0                 # 秒数


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

value_list = []  # 保存用配列

# Notify 呼び出し関数
def callback(sender, value):
    TimeStamp = time.time()
    shape_int16 = struct.unpack('>bbbbhhhhhh', value)

    value_acc_X, value_acc_Y, value_acc_Z = shape_int16[7], shape_int16[8], shape_int16[9]
    value_gyro_X, value_gyro_Y, value_gyro_Z = shape_int16[4], shape_int16[5], shape_int16[6]
    # データ保存
    value_list.append([TimeStamp, value_acc_X, value_acc_Y, value_acc_Z, value_gyro_X, value_gyro_Y, value_gyro_Z])
    # 表示
    print("Acc: {0} {1} {2}".format(value_acc_X, value_acc_Y, value_acc_Z))
    print("Gyro: {0} {1} {2}".format(value_gyro_X, value_gyro_Y, value_gyro_Z))

async def run(address, loop):
    async with BleakClient(address, loop=loop) as client:
        x = await client.is_connected()
        print("Connected: {0}".format(x))

        # サンプリング開始 20Hz
        await client.write_gatt_char(UUID7, bytearray([0x53, 0x17, 0x02, 0x01, 0x14]), response=True)

        await client.start_notify(UUID8, callback)

        # 5秒後に終了
        await asyncio.sleep(10.0, loop=loop)
        await client.stop_notify(UUID8)
        # サンプリング終了
        await client.write_gatt_char(UUID7, bytearray([0x53, 0x02, 0x02, 0x00, 0x00]), response=True)


def AddData():
    filename = "その他/value_list" + input("file_number 入力：") + ".csv"
    print(filename)
    address = (
        # discovery.pyでみつけたtoio Core Cubeのデバイスアドレスをここにセットする
        "00:04:79:00:0D:00"  # Windows か Linux のときは16進6バイトのデバイスアドレスを指定
        # if platform.system() != "Darwin"
        # else "243E23AE-4A99-406C-B317-18F1BD7B4CBE"  # macOSのときはmacOSのつけるUUID
    )
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run(address, loop))
    with open(filename, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(value_list)
