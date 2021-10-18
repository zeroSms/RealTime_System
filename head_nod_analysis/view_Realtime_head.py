import tkinter as tk
from time import sleep

index = 0  # 画像のindexはグローバルで管理する


def view_action():
    global index

    def btn_click(e):
        global index
        print(e.char)
        index = int(e.char)
        canvas.delete('p1')
        canvas.create_image(320, 213, image=photos[index], tag='p1')

    root = tk.Tk()
    root.geometry('700x560')
    root['bg'] = 'lightgrey'
    canvas = tk.Canvas(root, width=640, height=426, bd=0, highlightthickness=0, relief='ridge')
    canvas.pack(pady=20)
    photos = [
        tk.PhotoImage(file='キャプチャ0.PNG'),
        tk.PhotoImage(file='キャプチャ1.PNG'),
        tk.PhotoImage(file='キャプチャ2.PNG'),
    ]
    canvas.create_image(320, 213, image=photos[index], tag='p1')
    canvas.bind_all("<KeyPress>", btn_click)

    root.mainloop()


# ================================= メイン関数　実行 ================================ #
if __name__ == '__main__':
    view_action()
