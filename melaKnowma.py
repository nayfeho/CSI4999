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
    filename = filedialog.askopenfilename(initialdir="/images", title="Select Image",filetypes=[
                    (".jpg, .jpeg, .png, .bmp", ".jpeg"),
                    (".jpg, .jpeg, .png, .bmp", ".png"),
                    (".jpg, .jpeg, .png, .bmp", ".jpg"),
                    (".jpg, .jpeg, .png, .bmp", ".bmp"),
                ])
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
        prediction_label['text'] = f"Prediction: {'Benign ' if melanomapred < 0.50 else 'Melanoma '}"
        certainty_label['text'] = f"Certainty: {str(result) + '%'}"


window = tk.Tk()
NAME = "MelaKnowma"
DEFAULT_SIZE = "600x400"
window.title(NAME)
window.geometry(DEFAULT_SIZE)

prediction_label = tk.Label(window)
certainty_label = tk.Label(window)
#myLabel= Label(window, text= result ).place(x=50, y=50)
show_picture = tk.Label(window)
browse = tk.Button(window, text='Select Image',bg='#ffffff', fg='#000000',)

browse['command'] = selectPicture

prediction_label.pack(side=tk.TOP)
certainty_label.pack(side=tk.TOP)
show_picture.pack(side=tk.TOP)
browse.pack(side=tk.TOP)

window.mainloop()
