import numpy as np
import cv2 as cv
import tensorflow as tf
import os
from keras.models import load_model
import tkinter as tk
from tkinter import *
from tkinter import filedialog
import ttkbootstrap as ttk
from PIL import ImageTk, Image


def browser():
    filename = filedialog.askopenfilename(
        initialdir="/images", title="Select Image", filetypes=[
            ("image", "*.png *.jpg *.jpeg *.bmp")
        ])
    path.delete(0, END)
    path.insert(0, filename)
    set_photo()


def set_photo():
    global picture_path
    global img
    picture_path = path.get()
    img = Image.open(picture_path)
    img = img.resize((256, 256))
    img = ImageTk.PhotoImage(img)
    photo['image'] = img


def classify_photo():
    set_photo()
    # prediction_label['text'] = "Prediction: TEST"
    # certainty_label['text'] = "Certainty: 50%"
    if picture_path is not None:
        loadedmodel = load_model(os.path.join('models', 'modelV1.0'))

        melanomatestimg = cv.imread(picture_path)
        resizedtest2 = tf.image.resize(melanomatestimg, (256, 256))
        melanomapred = loadedmodel.predict(np.expand_dims(resizedtest2 / 255, 0))
        print(melanomapred)
        if melanomapred < 0.50:
            result = round(100 * (1 - melanomapred[0][0]), 2)
        else:
            result = round(100 * melanomapred[0][0], 2)
        prediction_label['text'] = f"Prediction: {'Benign ' if melanomapred < 0.50 else 'Melanoma '}"
        certainty_label['text'] = f"Certainty: {str(result) + '%'}"


window = tk.Tk()
NAME = "Mela-Know-ma"
DEFAULT_SIZE = "500x500"
window.title(NAME)
window.geometry(DEFAULT_SIZE)

path_label = ttk.Label(window, text='Photo Path: ')
path = ttk.Entry(window)
path.bind("<Return>", (lambda event: set_photo()))
browse_button = ttk.Button(window, text='Select Photo', command=browser)
photo = ttk.Label(window)
classify_button = ttk.Button(window, text='Identify', command=classify_photo)
prediction_label = ttk.Label(window)
certainty_label = ttk.Label(window)

path_label.pack()
path.pack()
browse_button.pack()
photo.pack()
classify_button.pack()
prediction_label.pack()
certainty_label.pack()

window.mainloop()
