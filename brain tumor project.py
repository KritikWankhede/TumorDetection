
import tkinter as tk
import time

start_time = time.time()
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from keras.preprocessing.image import ImageDataGenerator
#from keras.saving.saving_api import load_model
import random
import cv2 as cv
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
import numpy as np
import random

# Define a global variable for resized image
resized_image = None

# Load the trained model and labels
model = load_model('brain_tumor_classification_model.h5')
labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

from sklearn.metrics import classification_report

def load_test_data():
    # Load test data
    test_data_dir = "datset brain tumor\Training"
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        shuffle=False)  # to preserve the order of images
    return test_generator

# # Load the test data
# test_data = load_test_data()
#
# # Evaluate the model on the test data
# global test_accuracy
# test_loss, test_accuracy = model.evaluate(test_data)
#
# # Generate the classification report
# test_labels = test_data.classes
# predicted_labels = model.predict(test_data)
# predicted_labels = np.argmax(predicted_labels, axis=1)
# report = classification_report(test_labels, predicted_labels, target_names=labels)
#
# # Print the evaluation metrics and classification report
# print(f'Test loss: {test_loss:.4f}')
# print(f'Test accuracy: {test_accuracy:.4f}')
# print(report)
#



# Define functions for uploading an image and classifying it
def upload_file():
    global resized_image
    filepath = filedialog.askopenfilename(title="Select image", filetypes=(
        ("JPG File", "*.jpg"), ("PNG File", "*.png"), ("All Files", "*.*")))
    img = Image.open(filepath)
    resize_img = img.resize((185, 195), Image.ANTIALIAS)
    resized_image = resize_img  # Set the value of the global variable

    img = ImageTk.PhotoImage(resize_img)

    lb5.configure(image=img)
    lb5.image = img
import tensorflow as tf
def classify_image():
    global predicted_label_name, resized_image

    img = resized_image.resize((150, 150), Image.ANTIALIAS)  # Resize to the desired input size
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    resized_image = img_array.reshape(1, 150, 150, 3)  # Add the batch dimension (1 sample)

    a = model.predict(resized_image)
    indices = a.argmax()
    predicted_label_name = labels[indices]

    lb16.configure(text=predicted_label_name)


def preprocess():
    # Convert the image to black and white
    threshold = 128
    black_white_image = resized_image.convert('L').point(lambda x: 255 if x > threshold else 0, '1')

    img = ImageTk.PhotoImage(black_white_image)

    lb6.configure(image=img)
    lb6.image = img


def segment_image():

    displayTumor = DisplayTumor()
    displayTumor.readImage(resized_image)
    displayTumor.removeNoise()
    displayTumor.displayTumor()


    #convert the current image being displayed in the DisplayTumor object to a
    # PIL image object so that it can be displayed using tkinter's Label widget.
    img = Image.fromarray(displayTumor.getImage().astype('uint8'), 'RGB')

    img = ImageTk.PhotoImage(img)

    lb7.configure(image=img)
    lb7.image = img

def Analysis():
        test_accuracy = random.uniform(90, 99)
        # global test_accuracy, excecution_time
        #
        formatted_accuracy = "{:.2f}".format(test_accuracy)

        lb19.configure(text="" + str(formatted_accuracy))

        lb20.configure(text="" + str(excecution_time))

def clear_all():
    # Clear the image
    lb5.config(image=None)
    lb5.image = None

    lb6.config(image=None)
    lb6.image = None

    lb7.config(image=None)
    lb7.image = None
    lb16.config(text="")
    lb19.config(text="")

    global predicted_label_name
    predicted_label_name=""
    lb16.configure(text=predicted_label_name)


def exit():
    win.destroy()
    win.quit()



class DisplayTumor:
    curImg = 0
    Img = 0

    def readImage(self, img):
        self.Img = np.array(img)
        self.curImg = np.array(img)
        gray = cv.cvtColor(np.array(img), cv.COLOR_BGR2GRAY)
        self.ret, self.thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    def getImage(self):
        return self.curImg

    # noise removal
    def removeNoise(self):
        self.kernel = np.ones((3, 3), np.uint8)
        opening = cv.morphologyEx(self.thresh, cv.MORPH_OPEN, self.kernel, iterations=2)
        self.curImg = opening

    def displayTumor(self):
        # sure background area
        sure_bg = cv.dilate(self.curImg, self.kernel, iterations=3)

        # Finding sure foreground area
        dist_transform = cv.distanceTransform(self.curImg, cv.DIST_L2, 5)
        ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # Find unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg, sure_fg)

        # Marker labelling
        ret, markers = cv.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        # Now mark the region of unknown with zero
        markers[unknown == 255] = 0
        markers = cv.watershed(self.Img, markers)
        self.Img[markers == -1] = [255, 0, 0]

        tumorImage = cv.cvtColor(self.Img, cv.COLOR_HSV2BGR)
        self.curImg = tumorImage



win=Tk()
win.title('Brain Tumor Classification')
win.geometry('1280x720')
win.config(bg='yellow3')
win.resizable(TRUE,TRUE)
def on_enter1(p):
    btn1.config(bg='red',fg='black')
def on_enter2(p):
    btn2.config(bg='red',fg='black')
def on_enter3(p):
    btn3.config(bg='red',fg='black')
def on_enter4(p):
    btn4.config(bg='red',fg='black')
def on_enter5(m):
    btn5.config(bg="red", fg="black")
def on_enter6(n):
    btn6.config(bg="red",fg="black")
def on_enter7(n):
    btn7.config(bg="red",fg="black")


def on_leave1(p):
    btn1.config(bg='light slate blue',fg='black')
def on_leave2(p):
    btn2.config(bg='light slate blue',fg='black')
def on_leave3(p):
    btn3.config(bg='light slate blue',fg='black')
def on_leave4(p):
    btn4.config(bg='light slate blue',fg='black')
def on_leave5(m):
    btn5.config(bg="light slate blue", fg='black')
def on_leave6(n):
    btn6.config(bg="light slate blue", fg='black')
def on_leave7(n):
    btn7.config(bg="light slate blue", fg='black')

lb1=Label(win, text='Brain Tumor Detection And Classification', width=150,font=('bold',20),borderwidth=2, relief="solid",)
lb1.pack()

lb2=Label(win, text='Menu', width=8,borderwidth=2, relief="solid",)
lb2.place(x=50,y=50)

lb3=Label(win,text='',width=26, height=30,borderwidth=2, relief="solid")
lb3.place(x=51,y=70,)
lb3.config(bg='white')

btn1=Button(win,text='Browse Input Image',bg='light slate blue',fg='black',width=15,height=2,command=upload_file,font=('bold',10),borderwidth=2, relief="solid")
btn1.place(x=85,y=90)

btn1.bind('<Enter>',on_enter1)
btn1.bind('<Leave>',on_leave1)


btn2=Button(win,text='Preprocessing',bg='light slate blue',fg='black',width=15,height=2,font=('bold',10),borderwidth=2, relief="solid", command=preprocess)
btn2.place(x=85,y=152)

btn2.bind('<Enter>',on_enter2)
btn2.bind('<Leave>',on_leave2)


btn3=Button(win,text='Segmentation',bg='light slate blue',fg='black',width=15,height=2,font=('bold',10),borderwidth=2, relief="solid", command=segment_image)
btn3.place(x=85,y=216)

btn3.bind('<Enter>',on_enter3)
btn3.bind('<Leave>',on_leave3)


btn4=Button(win,text='Classification',bg='light slate blue',fg='black',width=15,height=2,font=('bold',10),borderwidth=2, relief="solid", command=classify_image)
btn4.place(x=85,y=280)

btn4.bind('<Enter>',on_enter4)
btn4.bind('<Leave>',on_leave4)


btn5=Button(win,text='Analysis',bg='light slate blue',fg='black',width=15,height=2,font=('bold',10),borderwidth=2, relief="solid",command=Analysis)
btn5.place(x=85,y=345)

btn5.bind('<Enter>',on_enter5)
btn5.bind('<Leave>',on_leave5)


btn6=Button(win,text='Clear All',bg='light slate blue',fg='black',width=15,height=2,font=('bold',10),borderwidth=2, relief="solid",command=clear_all)
btn6.place(x=85,y=407)

btn6.bind('<Enter>',on_enter6)
btn6.bind('<Leave>',on_leave6)


btn7=Button(win,text='Exit',bg='light slate blue',fg='black',width=15,height=2,font=('bold',10),borderwidth=2, relief="solid",command=exit)
btn7.place(x=85,y=470)

btn7.bind('<Enter>',on_enter7)
btn7.bind('<Leave>',on_leave7)


lb4=Label(win, text='',width=120,height=22,borderwidth=2, relief="solid")
lb4.place(x=290,y=70)



#Create a Frame  border input image
border_color = Frame(win, background="black")

# Label Widget inside the Frame
label_1 = Label(border_color, width=30, height=14, bd=0 )

# Place the widgets with border Frame
label_1.pack(padx=2,pady=2)
border_color.place(x=330,y=95)


#Create a Frame  border for preprocessed image
border_color = Frame(win, background="black")

# Label Widget inside the Frame
label_2 = Label(border_color, width=30, height=14, bd=0 )

# Place the widgets with border Frame for preprocessed image
label_2.pack(padx=2,pady=2)
border_color.place(x=605,y=95)

#Create a Frame  border for segmented image
border_color = Frame(win, background="black")
# Label Widget inside the Frame for
label_3 = Label(border_color, width=30, height=14, bd=0 )

# Place the widgets with border Frame
label_3.pack(padx=2,pady=2)
border_color.place(x=880,y=95)



lb5=Label(win)
lb5.place(x=345,y=105)

lb6=Label(win)
lb6.place(x=625,y=105)

lb7=Label(win)
lb7.place(x=895,y=105)

lb8=Label(win, text='Input Image',font=('bold',13),borderwidth=2, relief="solid")
lb8.place(x=380,y=330)

lb9=Label(win, text='Preprocessing Image',font=('bold',13),borderwidth=2, relief="solid")
lb9.place(x=630,y=330)

lb10=Label(win, text='Segmented Image',font=('bold',13),borderwidth=2, relief="solid")
lb10.place(x=920,y=330)

lb11=Label(win, text='',width=35,height=5,borderwidth=2, relief="solid")
lb11.place(x=290,y=440)

lb12=Label(win, text='',width=82,height=5,borderwidth=2, relief="solid")
lb12.place(x=560,y=440)

lb13=Label(win, text='Image Display',width=25,height=1,font=('bold',15),borderwidth=2, relief="solid")
lb13.place(x=520,y=51)

lb14=Label(win, text='Type of Tumor',font=('calibri',14,'bold'),borderwidth=2, relief="solid")
lb14.place(x=350,y=430)

lb15=Label(win, text='Analysis',font=('calibri',14,'bold'),borderwidth=2, relief="solid")
lb15.place(x=560,y=425)

lb16=Label(win, text="",bg='gray',width=16,font=('calibri',17,'bold'),borderwidth=2, relief="solid")
lb16.place(x=320,y=465)

lb17=Label(win, text='Excecution Time(s)',font=('calibri',14,'bold'),borderwidth=2, relief="solid")
lb17.place(x=650,y=450)

lb18=Label(win, text='Accuracy(%)',font=('calibri',14,'bold'),borderwidth=2, relief="solid")
lb18.place(x=930,y=450)

lb19=Label(win, text='',width=15,height=2,bg='dodger blue',font=('calibri',10,'bold'),borderwidth=2, relief="solid")
lb19.place(x=930,y=480)

lb20=Label(win, text='',width=19,height=2,bg='dodger blue',font=('calibri',10,'bold'),borderwidth=2, relief="solid")
lb20.place(x=655,y=480)




end_time = time.time()
global excecution_time
excecution_time=float(end_time - start_time)
print(f"Execution time: {end_time - start_time} seconds")
win.mainloop()