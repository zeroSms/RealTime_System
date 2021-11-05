import tkinter
from PIL import Image, ImageTk
import threading
import time

# 自作ライブラリ
from . import process_data, stop

global item, canvas
global img0, img1, img2
def show_image():
    # 外から触れるようにグローバル変数で定義
    global item, canvas
    global img0, img1, img2

    root = tkinter.Tk()
    root.title('test')
    root.geometry("300x300")

    # 切り替えたい画像を定義
    img0 = Image.open(r"../fig/head_neutral.PNG")
    img1 = Image.open(r"../fig/head_nod.PNG")
    img2 = Image.open(r"../fig/head_shake.PNG")
    img0 = ImageTk.PhotoImage(img0)
    img1 = ImageTk.PhotoImage(img1)
    img2 = ImageTk.PhotoImage(img2)

    canvas = tkinter.Canvas(bg="black", width=225, height=225)
    canvas.place(x=50, y=50)
    item = canvas.create_image(0, 0, image=img0, anchor=tkinter.NW)
    root.mainloop()


def view_action():
    if process_data.show_head == 1:
        canvas.itemconfig(item, image=img1)
    elif process_data.show_head == 2:
        canvas.itemconfig(item, image=img2)
    else:
        canvas.itemconfig(item, image=img0)


# ================================= メイン関数　実行 ================================ #
if __name__ == '__main__':
    show_image()
