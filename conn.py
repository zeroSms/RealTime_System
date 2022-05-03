conn = 90
byte_interval = bytearray \
    ([0x57, 0x00, 0x08, 0x03, 0xe8, 0x04, 0xb0, 0x00, 0x00, 0x00, 0x00])  # UUID7　書き込み用バイト（接続間隔）


# UUID7　書き込み用バイト（サンプリング開始）の周波数加算
def sampling_byte():
    # conn_min
    byte_interval[7] += int(int(conn / 1.25) >> 8 & 0b11111111)
    byte_interval[8] += int(int(conn / 1.25) & 0b11111111)

    # conn_max
    byte_interval[9] += int(int((conn+20) / 1.25) >> 8 & 0b11111111)
    byte_interval[10] += int(int((conn+20) / 1.25) & 0b11111111)

    # checkSum
    byte_interval[1] += sum(byte_interval[2:]) & 0b11111111

    return byte_interval


if __name__ == '__main__':
    print(bin(int(625 / 0.625)))
    print(bin(int(625 / 0.625) >> 8))
    print(hex(int(625 / 0.625) >> 8 & 0b11111111))
    print(hex(int(625 / 0.625) & 0b11111111))

    print(list(byte_interval))
    print(list(sampling_byte()))
