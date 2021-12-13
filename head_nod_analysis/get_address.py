#
# データ取得スレッド
#

import asyncio
from bleak import discover

eSense_name = ['eSense-0220', 'eSense-0328', 'eSense-0376', 'eSense-0459', 'eSense-0498', 'eSense-0586', 'eSense-0592', 'eSense-0666']

# ============================ eSenseのアドレスを取得 ============================== #
eSense_address = 0
async def search_eSense(eSense_number):
    global eSense_address
    eSense_flg = True
    while eSense_flg:
        devices = await discover()
        for d in devices:
            if eSense_name[eSense_number-1] in str(d):
                eSense_flg = False
                print(d)
                eSense_address = str(d).rsplit(':', 1)


# ============================ アドレス取得スレッド ============================== #
def Get(eSense_number):
    loop1 = asyncio.get_event_loop()
    loop1.run_until_complete(search_eSense(eSense_number))
    return eSense_address[0]
