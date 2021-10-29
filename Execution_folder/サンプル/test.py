import sys
import tkinter
from PIL import Image, ImageTk
import threading
import time
import random


def show_image():
    # 外から触れるようにグローバル変数で定義
    global item, canvas

    root = tkinter.Tk()
    root.title('test')
    root.geometry("300x300")
    img = Image.open(r"your_image.png")
    img = ImageTk.PhotoImage(img)
    canvas = tkinter.Canvas(bg="black", width=225, height=225)
    canvas.place(x=50, y=50)
    item = canvas.create_image(0, 0, image=img, anchor=tkinter.NW)
    root.mainloop()


# スレッドを立ててtkinterの画像表示を開始する
thread1 = threading.Thread(target=show_image)
thread1.start()

# 切り替えたい画像を定義
img1 = Image.open(r"your_image.png")
img2 = Image.open(r"your_image2.jpg")
img3 = Image.open(r"your_image3.png")
img1 = ImageTk.PhotoImage(img1)
img2 = ImageTk.PhotoImage(img2)
img3 = ImageTk.PhotoImage(img3)

list_ = [1, 2, 3]

while True:
    value = random.choice(list_)
    if value == 1:
        canvas.itemconfig(item, image=img1)
    elif value == 2:
        canvas.itemconfig(item, image=img2)
    elif value == 3:
        canvas.itemconfig(item, image=img3)
    time.sleep(1)
