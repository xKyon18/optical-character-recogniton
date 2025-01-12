from tkinter import *
from tkinter import messagebox
from tkinter.ttk import Progressbar
import tensorflow as tf
from PIL import ImageGrab, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import time

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(8, 3, input_shape=(28, 28, 1), activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])

BRUSH_SIZE = 12

def start_up():
    model.load_weights('cnn.h5')
    back.tkraise()

def draw(event):
    x = event.x
    y = event.y
    canvas.create_oval((x - BRUSH_SIZE / 2, y - BRUSH_SIZE / 2, 
                        x + BRUSH_SIZE / 2, y + BRUSH_SIZE / 2),
                        fill="white", outline="white")
    
def predict_canvas():
    x = window.winfo_rootx() + canvas.winfo_x()
    y = window.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()
    image = ImageGrab.grab().crop((x, y, x1, y1))
    image = ImageOps.fit(image, (280, 280))
    image = image.resize((28, 28))
    image = np.array(image, dtype=np.uint8)[:, :, 0]
    print(image.shape)
    image = (image / 255) - 0.5
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=3)
    pred = model.predict(image)
    res.set(f'Result: {str(np.argmax(pred, axis=1))}')

def clear_canvas():
    canvas.delete("all")
    res.set("Result: ")

window = Tk()
window.geometry('450x320')
window.title('Optical Character Recognition')
window.iconphoto(True, PhotoImage(file='Images/ocr.png'))
window.resizable(False, False)

front = Frame(window)
back = Frame(window)

front.grid(row=0, column=0, sticky=NSEW)
back.grid(row=0, column=0, sticky=NSEW)

####################################################################################################################################################################################################################

ocr = Label(front, text="Optical Character Recognition", font=('Tahoma', 19, 'bold'))
ocr.grid(row=0, column=0, columnspan=2, padx=30, pady=10)

desc = Label(front, text="This application leverages a Convolutional Neural Network (CNN) with\n with the help of TensorFlow framework to predict handwritten digits\n drawn by the user on an interactive canvas.",
             font=('Tahoma', 10), justify='center')
desc.grid(row=1, column=0, columnspan=2) 

start = Button(front, text="start", width=10, command=start_up, font=('Tahoma', 10))
start.grid(row=3, column=0, columnspan=2, pady=20)

####################################################################################################################################################################################################################

canvas = Canvas(back, height="280", width="280", bg="#000000", bd=0, highlightthickness=0)
canvas.grid(row=0, column=0, padx=20, pady= 20, columnspan=4, rowspan=20)
canvas.bind('<B1-Motion>', draw)

predict = Button(back, text='Predict', width=10, command=predict_canvas, font=('Tahoma', 10))
predict.grid(row=6, column=4, sticky=W)

res = StringVar()
res.set("Result: ") 
result = Label(back, textvariable=res, width=14, font=('Tahoma', 10), bd=3, anchor=W)
result.grid(row=8, column=4, columnspan=2, sticky=W)

clear = Button(back, text='Clear', width=10, command=clear_canvas, font=('Tahoma', 10))
clear.grid(row=15, column=4, sticky=W)

exit = Button(back, text='Exit', width=10, command=quit, font=('Tahoma', 10))
exit.grid(row=16, column=4, sticky=W)

front.tkraise()
window.mainloop()