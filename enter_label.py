#
# 正解ラベル入力スレッド
#

from pynput import keyboard

# ラベル判別用変数
label_flg = "others"  # プログラム終了用フラグ

# ================================= コールバック関数 ================================ #
# キーを押している間
def press(key):
    global label_flg
    try:
        if key.char == '1':
            label_flg = "nod"       # うなずく
        elif key.char == '2':
            label_flg = "shake"     # 首振り

    # 特殊キーが押されたとき
    except AttributeError:
        pass

# キーを離したとき
def release(key):
    global label_flg
    label_flg = "others"
    if key == keyboard.Key.enter:
        return False


# ============================ 正解ラベル入力スレッド ============================== #
def Label():
    global label_flg

    # enterで終了
    with keyboard.Listener(on_press=press, on_release=release) as listener:
        listener.join()

