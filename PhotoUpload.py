import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image

def selectPicture():
    global img
    filename = filedialog.askopenfilename(initialdir="/images", title="Select Image",filetypes=(("png images","*.png"),("jpg images","*.jpg"),("jpeg images","*.jpeg"),("bmp images","*.bmp")))
    img = Image.open(filename)
    img = img.resize((200,200))
    img = ImageTk.PhotoImage(img)
    show_picture['image'] = img
    picture_path = filename

window = tk.Tk()
NAME = "MelaKnowma"
DEFAULT_SIZE = "600x400"
window.title(NAME)
window.geometry(DEFAULT_SIZE)

show_picture = tk.Label(window)
browse = tk.Button(window, text='Select Image',bg='grey', fg='#ffffff',)

browse['command'] = selectPicture

show_picture.place(x=190, y= 75)
browse.place(x=250, y=350)


window.mainloop()

