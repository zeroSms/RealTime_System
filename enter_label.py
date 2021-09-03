#
# 正解ラベル入力スレッド
#

import keyboard
import stop


# ラベル判別用変数
label_flg = "others"  # プログラム終了用フラグ

# ============================ 正解ラベル入力スレッド ============================== #
def Label():
    global label_flg

    # enterで終了
    while stop.stop_flg:
        key_num = keyboard.read_key()
        if   key_num == "1":    label_flg = "nod"       # うなずく
        elif key_num == "2":    label_flg = "shake"     # 首振り
        else:                   label_flg = "others"    # その他