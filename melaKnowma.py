import numpy as np
import cv2 as cv
import tensorflow as tf
import os
from keras.models import load_model
import tkinter as tk
from tkinter import *
from tkinter import ttk, filedialog
from PIL import ImageTk, Image


def browser():
    filename = filedialog.askopenfilename(
        initialdir='/images', title='Select Image', filetypes=[
            ('image', '*.png *.jpg *.jpeg *.bmp')
        ])
    # Don't continue if file was not picked (i.e. user closed browser)
    if filename != '':
        # Reset path entry to new file path
        path.delete(0, END)
        path.insert(0, filename)
        set_photo()


def set_photo():
    global picture_path
    global img

    error_label['text'] = ''
    error_label['background'] = '#f0f0f0'
    result_label['text'] = ''
    percent_label['text'] = ''

    picture_path = path.get()
    # Show rightmost side of path if too large to fit
    path.xview_moveto(1)
    try:
        img = Image.open(picture_path)
        img = img.resize((256, 256))
        img = ImageTk.PhotoImage(img)
        photo_label['text'] = 'Your Photo:'
        photo['image'] = img
        return True
    except (FileNotFoundError, AttributeError):
        photo_label['text'] = 'Example Photo:'
        photo['image'] = default_img
        error_label['text'] = 'File not found. Please try a different path.'
        error_label['background'] = 'brown2'
        return False


def classify_photo():
    good_path = set_photo()
    if good_path:
        loadedmodel = load_model(os.path.join('models', 'modelV1.0'))

        melanomatestimg = cv.imread(picture_path)
        resizedtest2 = tf.image.resize(melanomatestimg, (256, 256))
        melanomapred = loadedmodel.predict(np.expand_dims(resizedtest2 / 255, 0))
        print(melanomapred)
        if melanomapred < 0.50:
            result = round(100 * (1 - melanomapred[0][0]), 2)
        else:
            result = round(100 * melanomapred[0][0], 2)
        result_label['text'] = f"{'BENIGN' if melanomapred < 0.50 else 'MELANOMA'}"
        percent_label['text'] = f"{str(result)}%"


# Window
window = tk.Tk()
NAME = 'Mela-Know-ma'
DEFAULT_SIZE = '600x500'
window.title(NAME)
window.geometry(DEFAULT_SIZE)

default_img = Image.open('./melanoma-picture.png').resize((256, 256))
default_img = ImageTk.PhotoImage(default_img)

# Frames
path_frame = ttk.Frame(window)
ident_frame = ttk.Frame(window)

# Widgets
path_label = ttk.Label(window, text='Photo Path:', font=('Arial', 11))
path = ttk.Entry(window)
# Reset photo on pressing enter key inside path entry
path.bind('<Return>', (lambda event: set_photo()))
browse_button = ttk.Button(window, text='Select Photo', command=browser)
type_label = ttk.Label(window, text='(.png, .jpg, .jpeg, .bmp)')

identify_button = ttk.Button(ident_frame, text='Identify', command=classify_photo)
error_label = ttk.Label(ident_frame)

photo_label = ttk.Label(window, text='Example Photo:', font=('Arial', 12))
photo = ttk.Label(window, image=default_img)

prediction_label = ttk.Label(window, text='Result:', font=('Arial', 13))
result_label = ttk.Label(window, font=('Arial', 19, 'bold'))
certainty_label = ttk.Label(window, text='Certainty:', font=('Arial', 13))
percent_label = ttk.Label(window, font=('Arial', 22, 'bold'))

# Grids
window.rowconfigure(tuple(range(15)), weight=1, minsize=25, uniform='a')
window.columnconfigure(tuple(range(10)), weight=1, minsize=55, uniform='a')

# Top Layout
path_label.grid(sticky='sw', row=0, column=0, columnspan=10, padx=7)
path.grid(sticky='ew', row=1, column=0, columnspan=4, padx=7)
browse_button.grid(sticky='w', row=1, column=4, columnspan=5)
type_label.grid(sticky='nw', row=2, columnspan=9, padx=7)
identify_button.pack(side='left', padx=7, fill='both')
error_label.pack(side='left', padx=7, fill='both', expand=True)
ident_frame.grid(sticky='nsew', row=4, column=0, columnspan=8)

# Left Layout
photo_label.grid(sticky='sw', row=6, column=0, columnspan=3, padx=7)
photo.grid(sticky='nsew', row=7, column=0, rowspan=9, columnspan=11, padx=7)

# Right Layout
prediction_label.grid(sticky='w', row=8, column=5, columnspan=5, padx=7)
result_label.grid(sticky='sw', row=9, column=5, columnspan=5, padx=7)
certainty_label.grid(sticky='nsew', row=11, column=5, columnspan=5, padx=7)
percent_label.grid(sticky='nsew', row=12, column=5, columnspan=5, padx=7)

# Run
window.mainloop()
