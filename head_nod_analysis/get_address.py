#
# データ取得スレッド
#

import asyncio
from bleak import discover

# ============================ eSenseのアドレスを取得 ============================== #
eSense_address = 0
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


# ============================ アドレス取得スレッド ============================== #
def Get():
    loop1 = asyncio.get_event_loop()
    loop1.run_until_complete(search_eSense())
    return eSense_address[0]
