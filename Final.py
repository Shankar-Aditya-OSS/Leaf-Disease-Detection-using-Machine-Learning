from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import os
import cv2
import matplotlib.pyplot as plt
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import load_model
import numpy as np
from tensorflow import keras
import pickle
from PIL import ImageTk,Image  
import mysql.connector
import sqlite3
import webbrowser


conn = sqlite3.connect('LDD.db')
mycursor = conn.cursor()
#-------------------------------------------------------- 

def additional():
  def onclick():
    pass
  root = Tk()
  root.geometry('975x600')
  root.resizable(False, False)
  root.config(bg='SpringGreen2')
  def add_window(x,y,content,pack_window):
    #White space for displaying text
    text = Text(root,height=25,width=55,wrap = WORD,padx=12, pady=12)
    
    #for vertical scrollbar(in-window)
    v = Scrollbar(root, orient='vertical')
    v.config(command=text.yview)
    text.insert(END, content)
    text.place(x=x,y=y)
    text.configure(state='disabled')
    text.pack(side=pack_window,padx=10)
    
  #To display content
  query = "SELECT leaf_name,disease_name,description,cure,links FROM Disease where leaf_name='"+diseasename2[0]+"' and disease_name='"+diseasename2[1]+"'"
  mycursor.execute(query)
  myresult = mycursor.fetchall()

  for x in myresult:
    leaf=x[0]
    diseaseleaf=x[1]
    description=x[2]
    treatment=x[3]
    link=x[4]
  titlename=leaf+" Category - "+diseaseleaf+" Disease"
  root.title(titlename)
  
  lbl = Label(root, text="Description:",width=13  ,fg="black" ,height=2 ,font=('times', 15, ' bold')) 
  lbl.place(x=10, y=15)  
  add_window(10,0,description,tkinter.LEFT)

  lb2 = Label(root, text="Cure:",width=13  ,fg="black",height=2 ,font=('times', 15, ' bold')) 
  lb2.place(x=500, y=15)
  add_window(400,0,treatment,tkinter.RIGHT)

  lb3 = Label(root, text="For More Info :",width=12  ,fg="black",height=1 ,font=('times', 13, ' bold')) 
  lb3.place(x=10, y=545)

  link1 = Label(root, text=link, fg="blue", cursor="hand2",font=('times',13,'bold'))
  link1.place(x=145,y=545)
  link1.bind("<Button-1>", lambda e: webbrowser.open_new(link))

  root.mainloop()

#--------------------------------------------------------
main = tkinter.Tk()
main.title("Leaf Disease Detection")
main.geometry("950x600")
main.resizable(False,False)
statusmodel=False

message_error = Label(main, text=""  ,fg="red"  ,width=25 ,height=1,bg="SpringGreen2"  ,font=('times', 12, ' bold ')) 
message_error.place(x=360, y=563)

try:
    model = load_model('save')
    filename = 'plant_disease_label_transform.pkl'
    image_labels = pickle.load(open(filename, 'rb'))
    print("Model Loaded Successfully")
    message_error.configure(text="Model Loaded Successfully")
    statusmodel=True
    
except:
    print("ERROR")
    message_error.configure(text="Could Not Load the Model")
    statusmodel=False


# Dimension of resized image
DEFAULT_IMAGE_SIZE = tuple((256, 256))

def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None:
            image = cv2.resize(image, DEFAULT_IMAGE_SIZE)   
            return img_to_array(image)
        else:
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None
    
def predict_disease(image_path):
    image_array = convert_image_to_array(image_path)
    np_image = np.array(image_array, dtype=np.float16) / 225.0
    np_image = np.expand_dims(np_image,0)
    plt.imshow(plt.imread(image_path))
    result = model.predict_classes(np_image)
    return image_labels.classes_[result][0]

imagename=[]
diseasename2=[]

def upload():
    global filename
    imagename.clear()
    filename = askopenfilename(initialdir = r"D:\Major-Project\Leaf Disease Detection\LDD\PlantVillage")
    pathlabel.config(text=filename)
    imagename.append(filename)  
    image = Image.open(filename)
    resize_image = image.resize((450, 450))
    img = ImageTk.PhotoImage(resize_image)
    canvas.create_image(2, 2, anchor=NW, image=img)
    pt.config(state='normal')
    main.mainloop()

def predict():
    diseasename=predict_disease(imagename[0])
    diseasename = diseasename.replace("_", " ")
    diseasename1 = diseasename.split(",")
    diseasename2.clear()
    diseasename2.append(diseasename1[0])
    diseasename2.append(diseasename1[1])
    message10.configure(text=diseasename1[0])
    if(diseasename1[1]=='Healthy'):
      info.config(state='disabled')
      message20.configure(text='No Disease - Healthy')
    else:
      info.config(state='normal')
      message20.configure(text=diseasename1[1])
    
    
font = ('times', 16, 'bold')
title = Label(main, text='Leaf Disease Detection')
title.config(fg='black')  
title.config(font=font)           
title.config(height=3, width=77)       
title.place(x=10,y=7)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Image", command=upload)
upload.place(x=550,y=130)
upload.config(font=font1)
if statusmodel:
  upload.config(state='normal')
else:
  upload.config(state='disabled')

pathlabel = Label(main)

pt = Button(main, text="Predict", command=predict)
pt.place(x=730,y=130)
pt.config(font=font1,state='disabled')

info = Button(main, text="Show Info", command=additional)
info.place(x=650,y=450)
info.config(font=font1,state='disabled')

lbl = Label(main, text="Leaf Category",width=13  ,fg="black"  ,height=2 ,font=('times', 15, ' bold')) 
lbl.place(x=480, y=250)
lb2 = Label(main, text="Disease",width=13  ,fg="black"  ,height=2 ,font=('times', 15, ' bold')) 
lb2.place(x=480, y=350)

message10 = Label(main, text="" ,fg="black"  ,width=23 ,height=2, activebackground = "yellow" ,font=('times', 15, ' bold ')) 
message10.place(x=660, y=250)
message20 = Label(main, text="" ,fg="black"  ,width=23 ,height=2, activebackground = "yellow" ,font=('times', 15, ' bold ')) 
message20.place(x=660, y=350)

canvas = Canvas(main, width = 450, height = 450)
canvas.delete('all')
canvas.pack()
canvas.place(x=10, y=100)
IMage = Image.open(r"D:\Major-Project\Leaf Disease Detection\LDD\PlantVillage\thumbnail.png")
resize_image = IMage.resize((450, 450))
img = ImageTk.PhotoImage(resize_image)
canvas.create_image(2, 2, anchor=NW, image=img)

main.config(bg='teal')
main.mainloop()



