import numpy as np 
import cv2 as cv
import matplotlib as plt
from matplotlib import pyplot as plt
import tensorflow as tf
import os
from keras import datasets, layers, models
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.metrics import Precision, Recall, BinaryAccuracy
import imghdr
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image

def selectPicture():
    global img
    global picture_path
    filename = filedialog.askopenfilename(initialdir="/images", title="Select Image",filetypes=(("png images","*.png"),("jpg images","*.jpg"),("jpeg images","*.jpeg"),("bmp images","*.bmp")))
    img = Image.open(filename)
    img = img.resize((200,200))
    img = ImageTk.PhotoImage(img)
    show_picture['image'] = img
    picture_path = filename
    if picture_path is not None:
        loadedmodel = load_model(os.path.join('models', 'modelV1.0'))

        melanomatestimg = cv.imread(picture_path)
        resizedtest2 = tf.image.resize(melanomatestimg, (256, 256))
        melanomapred = loadedmodel.predict(np.expand_dims(resizedtest2/255, 0))
        print(melanomapred)
        if melanomapred < 0.50:
            result = round(100*(1-melanomapred[0][0]), 2)
        else:
            result = round(100*melanomapred[0][0], 2)
        result_label['text'] = f"Prediction: {'Benign ' + str(result) + '% Certainty' if melanomapred < 0.50 else 'Melanoma ' + str(result) + '% Certainty'}"


window = tk.Tk()
NAME = "MelaKnowma"
DEFAULT_SIZE = "600x400"
window.title(NAME)
window.geometry(DEFAULT_SIZE)

result_label = tk.Label(window, text="Prediction: ")
#myLabel= Label(window, text= result ).place(x=50, y=50)
show_picture = tk.Label(window)
browse = tk.Button(window, text='Select Image',bg='grey', fg='#ffffff',)

browse['command'] = selectPicture

show_picture.place(x=190, y= 75)
browse.place(x=250, y=350)
result_label.place(x=250, y=20)

window.mainloop()
