#!/usr/bin/env python
# coding: utf-8

# ### Installing Required Packages

# In[1]:


pip install tk Pillow pandas numpy python-math opencv-contrib-python random2 joblib keras-models Pillow zipfile36 pathlib2 matplotlib


# In[2]:


pip list


# ### Importing required packages

# In[3]:


import tkinter as tk
import tkinter.font as font
from numpy import argmax
import numpy as np
import os
import math
import cv2
import pandas as pd
import random
import joblib
from keras.models import load_model
from PIL import ImageTk, Image
import zipfile
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ### Setting Variables and functions

# In[4]:


Width = 1080
Height = 720

#Function that closes a window and opens a new one.
def CloseAndOpen(Close,Type):
    Close.destroy()
    if Type == "Model Results":
        OpenModelingResults()
    elif Type == "Image Details":
        OpenImageDetails()
    elif Type == "Guessing Game":
        OpenGuessingGame()
    elif Type == "Main Menu":
        OpenMainMenu()
    elif Type == "Help":
        OpenUserGuide()

#Resizes an image        
def Resize(image_,w,h):
    newsize = (w, h)
    return cv2.resize(image_, dsize = newsize)        

#Same ColorFiltering function used in the modeling. Creates filters on the image.
def ColorFilters(fishImage):
    RedF = (fishImage[:,:,0] > 100)
    InvBlueF = (fishImage[:,:,2] < 115)
    WhiteF = ((fishImage[:,:,0] > 180) & (fishImage[:,:,1] > 180) & (fishImage[:,:,2] > 130))
    BlackF = ((fishImage[:,:,0] < 25) & (fishImage[:,:,1] < 25) & (fishImage[:,:,2] < 25))
    GrayF = (fishImage[:,:,0] < 140) & (fishImage[:,:,1] < 140) & (fishImage[:,:,2] < 140) & (fishImage[:,:,0] > 100) & (fishImage[:,:,1] > 100) & (fishImage[:,:,2] > 100)
    DGrayF = (fishImage[:,:,0] < 100) & (fishImage[:,:,1] < 100) & (fishImage[:,:,2] < 100) & (fishImage[:,:,0] > 80) & (fishImage[:,:,1] > 80) & (fishImage[:,:,2] > 80)
    Filter = (RedF*InvBlueF) + (GrayF + DGrayF + WhiteF + BlackF)

    FilteredImage = fishImage.copy()
    FilteredImage[:, :, 0] = FilteredImage[:, :, 0] * Filter
    FilteredImage[:, :, 1] = FilteredImage[:, :, 1] * Filter
    FilteredImage[:, :, 2] = FilteredImage[:, :, 2] * Filter
    return FilteredImage        
 
# Same function used in the modeling. This function cateogrizes each pixel into one of the
# colors below. Background colors are reclassifed as black and the colors are totaled to get the
# number of pixels for each color
def ClassifyPixelColor(Image):
    ColorLabels = ['Red','Orange','Brown','Tan','Blue', 'LightBlue','Gray', 'White','Black']
    ColorValues = [[200,75,75], [200,125,75], [151,122,83], [217,171,118],[75,75,200],[165,165,165],[125,125,125],[235,235,235],[20,20,20]]
    ColorValuesMap = [[200,75,75], [200,125,75], [151,122,83], [217,171,118],[20,20,20],[20,20,20],[20,20,20],[235,235,235],[20,20,20]]
    w,h,d = Image.shape 
    zeros = np.zeros(9, dtype=int)
    dim = (w,h)
    totals = []
    ModifiedImages = []
    Reshaped = np.reshape(Image,(w*h,3))
    ColorLabels = np.array(ColorLabels)
    zeros = np.zeros(9, dtype=int)
    for p in range(w*h):
        Dif = np.array(np.sum(abs(Reshaped[p]-ColorValues), axis = 1))
        Current = 1000
        Color=''
        for c in range(len(ColorLabels)):
            if Dif[c] < Current:
                Color = ColorValuesMap[c]
                Current = Dif[c]
        zeros[ColorValuesMap.index(Color)] = zeros[ColorValuesMap.index(Color)] + 1
        Reshaped[p] = Color
    totals.append(zeros)
    ModifiedImages.append(np.reshape(Reshaped,(h,w,3)))
    totals = np.array(totals)
    totals = pd.DataFrame(totals)
    totals.columns = ["Reds", "Oranges", "Browns", "Tans", "Blues", "LightBlues", "Grays", "Whites", "Blacks"]
    totals = totals[["Reds", "Oranges", "Browns", "Tans", "Grays","Whites"]]
    return totals, np.array(ModifiedImages)


# ### Main Menu

# In[5]:


# This function creates the main menu window.

def OpenMainMenu():
      
    #Create an instance of Tkinter frame or window
    MainWin= tk.Tk()
    MainWin.title("Main Menu")
    MainWin.geometry(str(Width)+'x'+str(Height))
    #Set the geometry of tkinter frame

    myFont = font.Font(size=16, weight='bold')

    #Create a Label and a Button widget
    Titlelb = tk.Label(MainWin, text="", font=myFont).pack(pady=4)
    Titlelb = tk.Label(MainWin, text="Classifying Images of 9 different seafood", font=myFont).pack(pady=4)
    
    MRbtn = tk.Button(MainWin, 
                      text="Modeling Results", 
                      command = lambda:CloseAndOpen(MainWin,"Model Results"), 
                      width=15, 
                      height=2,
                      font = myFont,
                      bg='#0052cc', 
                      fg='#ffffff')
    
    IDbtn = tk.Button(MainWin, 
                      text="Image Details", 
                      command= lambda:CloseAndOpen(MainWin,"Image Details"), 
                      width=15, 
                      height=2, 
                      font = myFont,
                      bg='#0052cc', 
                      fg='#ffffff')
    
    GGbtn = tk.Button(MainWin,
                      text="Guessing Game", 
                      command= lambda:CloseAndOpen(MainWin,"Guessing Game"),  
                      width=15, 
                      height=2, 
                      font = myFont,
                      bg='#0052cc', 
                      fg='#ffffff')
    Helpbtn = tk.Button(MainWin,
                      text="User's Guide", 
                      command= lambda:CloseAndOpen(MainWin,"Help"),  
                      width=15, 
                      height=2, 
                      font = myFont,
                      bg='#0052cc', 
                      fg='#ffffff')

    #placing widgets
    MRbtn.place(relx=0.5, rely=0.2, anchor=tk.CENTER)
    IDbtn.place(relx=0.5, rely=0.4, anchor=tk.CENTER)
    GGbtn.place(relx=0.5, rely=0.6, anchor=tk.CENTER)
    Helpbtn.place(relx = .5, rely = .8, anchor=tk.CENTER)


    MainWin.bind('<Return>',lambda event:callback())
    MainWin.mainloop()


# ### Modeling Results

# In[6]:


# This function creates the Modeling results window

def OpenModelingResults():
    #Create an instance of Tkinter frame or window
    ResultsWin= tk.Tk()
    ResultsWin.title("Modeling Results")    
    ResultsWin.geometry(str(Width)+'x'+str(Height))

    # Varibles
    myFont = font.Font(size=15, weight='bold')
    ModelOptions = ["Convolutional Neural Networ (CNN)","K-Nearest Neighbors (KNN)", "Support Vector Machine (SVM)", "Random Forest (RF)"]
    ModelSelected = tk.StringVar(ResultsWin,"")
    ModelSelected.set("Convolutional Neural Networ (CNN)")

    #Functions
    def SwapConfusionMatrix():
        if ModelSelected.get() == "Convolutional Neural Networ (CNN)":
            CF_img = "ConfusionMatrix_CNN.png"
        elif ModelSelected.get() == "K-Nearest Neighbors (KNN)":
            CF_img = "ConfusionMatrix_KNN.png"
        elif ModelSelected.get() == "Support Vector Machine (SVM)":
            CF_img = "ConfusionMatrix_SVM.png"
        else:
            CF_img = "ConfusionMatrix_RF.png"
        CF_img = ImageTk.PhotoImage(file = CF_img)
        CMLabel.configure(image = CF_img)
        CMLabel.image = CF_img 
        return

    #Create labels, buttons, images, and dropdowns
    Backbtn = tk.Button(ResultsWin, 
                        text="Back", 
                        command= lambda:CloseAndOpen(ResultsWin,"Main Menu"),
                        width=15, 
                        height=1, 
                        bg='#0052cc', 
                        fg='#ffffff')
    GetCMButton = tk.Button(ResultsWin, 
                        text="Fetch Confusion Matrix", 
                        command= SwapConfusionMatrix,
                        height=1, 
                        bg='#0052cc', 
                        fg='#ffffff')
    drop = tk.OptionMenu( ResultsWin , ModelSelected , *ModelOptions )
    CMLabel = tk.Label(ResultsWin, image ="")
    
    DataTable = tk.Label(ResultsWin,image = "")
    DataTableImage = ImageTk.PhotoImage(file = "TimeTable.png")
    DataTable.configure(image = DataTableImage)
    DataTable.image = DataTableImage 
    
    ROC = tk.Label(ResultsWin,image = "")    
    ROCImage = ImageTk.PhotoImage(file = "ROC-AUC.png")
    ROC.configure(image = ROCImage)
    ROC.image = ROCImage     
    SwapConfusionMatrix() #Gets the Initial Confusion Matrix 

    
    #Placing widgets
    drop.pack()
    GetCMButton.pack()
    CMLabel.place(relx=.3, rely = .3, anchor=tk.CENTER)
    ROC.place(relx=.7, rely = .3, anchor=tk.CENTER)
    DataTable.place(relx = .5, rely = .7, anchor = tk.CENTER)
    Backbtn.place(relx=0.5, rely=0.85, anchor=tk.CENTER)

    ResultsWin.bind('<Return>',lambda event:callback())
    ResultsWin.mainloop()


# ### Image Details

# In[7]:


# Image Detail Function creates the Image Detail Window

def OpenImageDetails():
    #Create an instance of Tkinter frame or window
    ImageDetailsWin= tk.Tk()
    ImageDetailsWin.title("Image Details")    
    ImageDetailsWin.geometry(str(Width)+'x'+str(Height))
    
# Variables       
    myFont = font.Font(size=15, weight='bold')
    options = ["Black Sea Sprat", "Gilt-Head Bream", "Hourse Mackerel", "Red Mullet", "Red Sea Bream", "Sea Bass", "Shrimp","Striped Red Mullet", "Trout"]
    ImagePath = os.path.join(os.getcwd(),'archive.zip','Fish_Dataset','Fish_Dataset')  
    #ErrorMessage = tk.StringVar(ImageDetailsWin)
    #ErrorMessage.set("")

# Functions
    def GetImage(initial):
        img_ = []    
        PixelColorCounts = []
        if initial == 1:
            ImageNumber = str(random.randint(1, 9000))
            ErrorLabel.config( text = "Please enter a number between 1 and 9,000." )
        else:
            ImageNumber = ImageEntry.get()
        if ImageNumber.isdigit() == True:
            ImageNumber = int(ImageNumber)
            if ImageNumber > 0 and ImageNumber <=9000:
                #ErrorLabel.config( text = ImageNumber )
                fishfolder = options[math.floor((ImageNumber-1)/1000)]
                Img_num = ImageNumber - math.floor((ImageNumber-1)/1000)*1000
                if Img_num < 10:
                    Img_num = '0000'+str(Img_num)+'.png'
                elif Img_num < 100:
                    Img_num = '000'+str(Img_num)+'.png'
                elif Img_num < 1000:
                    Img_num = '00'+str(Img_num)+'.png'      
                else:
                    Img_num = '0'+str(Img_num)+'.png'   
                UpdatedImagePath = os.path.join(ImagePath,fishfolder,fishfolder,Img_num)
                
                with zipfile.ZipFile(os.path.join(os.getcwd(),'archive.zip'), 'r') as zipref:
                    for imagepath in zipref.namelist():
                        if (imagepath.__contains__('GT')):
                            pass
                        elif (imagepath.__contains__('NA_Fish_Dataset')):
                            pass
                        elif (imagepath.__contains__(str("/"+fishfolder+"/"+fishfolder+"/"+Img_num))): #Text was img
                            img_.append(Image.open(zipref.open(imagepath)))
                Fish_Image = img_[0]  
                
                #Get Model Labels
                CNNprediction = load_model('CNNModel.h5').predict(np.array([np.array(cv2.resize(np.array(Fish_Image),dsize = (75,75)))]))
                CNNyhat = argmax(CNNprediction, axis=-1).astype('int')
                pred = options[CNNyhat[0]]
                ActualLabel.config(text = "Actual Label: "+ str(fishfolder))
                CNNLabel.config(text = "CNN Prediction: " +str(pred))                
                
                ResizeImage = Resize(np.array(Fish_Image),50,50)
                counts, ModifiedImage = ClassifyPixelColor(ResizeImage)

                KNNpredict = joblib.load('KNNModel.joblib').predict(counts)
                KNNLabel.config(text = "KNN Prediction: " +str(options[KNNpredict[0]]))   
                
                SVMpredict = joblib.load('SVMModel.joblib').predict(counts)
                SVMLabel.config(text = "SVM Prediction: " +str(options[SVMpredict[0]])) 
                
                RFpredict = joblib.load('RFModel.joblib').predict(counts)
                RFLabel.config(text = "RF Prediction: " +str(options[RFpredict[0]])) 
                
                Fish_Image = Resize(np.array(Fish_Image), 225, 225)
                ResizeImage = Resize(Resize(np.array(Fish_Image),50,50), 225,225)
                ModifiedImage = Resize(ModifiedImage[0], 225,225)

                
                Fish_Image = np.array(Fish_Image).astype("uint8")
                Fish_Image = ImageTk.PhotoImage(image=Image.fromarray(Fish_Image))
                ResizeImage = np.array(ResizeImage).astype("uint8")
                ResizeImage = ImageTk.PhotoImage(image=Image.fromarray(ResizeImage))
                ModifiedImage = np.array(ModifiedImage).astype("uint8")
                ModifiedImage = ImageTk.PhotoImage(image=Image.fromarray(ModifiedImage))
                
                OriginalImageLabel.configure(image = Fish_Image)
                OriginalImageLabel.image = Fish_Image 
                ResizeImageLabel.configure(image = ResizeImage)
                ResizeImageLabel.image = ResizeImage 
                PostProccessedImage.configure(image = ModifiedImage)
                PostProccessedImage.image = ModifiedImage 

                
            else:
                ErrorLabel.config( text = "Please enter a number between 1 and 9,000." )
        else:
            ErrorLabel.config( text = "Please enter a number between 1 and 9,000." )    
        
    
# Window Elements
    ImageEntry = tk.Entry(ImageDetailsWin, textvariable = 125,bd = 5)
    
    Backbtn = tk.Button(ImageDetailsWin, 
                        text="Back", 
                        command= lambda:CloseAndOpen(ImageDetailsWin,"Main Menu"),
                        width=15, 
                        height=1, 
                        bg='#0052cc', 
                        fg='#ffffff')
    GetImageButton = tk.Button(ImageDetailsWin, 
                        text="Get Image", 
                        command= lambda:GetImage(0),
                        width=15, 
                        height=1, 
                        bg='#0052cc', 
                        fg='#ffffff')
    ErrorLabel = tk.Label( ImageDetailsWin, text = "" )
    OriginalImageLabel = tk.Label(ImageDetailsWin, image ="")
    ResizeImageLabel = tk.Label(ImageDetailsWin, image ="")
    PostProccessedImage = tk.Label(ImageDetailsWin, image ="")
    OriginalText = tk.Label(ImageDetailsWin, text = "Original Image")
    ProcessedText = tk.Label(ImageDetailsWin, text = "Removed Background Colors")
    ResizeText = tk.Label(ImageDetailsWin, text = "Resized and Blurred")
     
        
    ActualLabel = tk.Label(ImageDetailsWin, text = "")
    CNNLabel = tk.Label(ImageDetailsWin, text = "")   
    KNNLabel = tk.Label(ImageDetailsWin, text = "")   
    SVMLabel = tk.Label(ImageDetailsWin, text = "")   
    RFLabel = tk.Label(ImageDetailsWin, text = "")   
    
    GetImage(1)
        
    # Placing Widgets
    GetImageButton.place(x = 585, y = 25)
    ImageEntry.place(x = 450, y = 25)
    ErrorLabel.place(x = 450, y = 55)
    
    CNNLabel.place(x = 710, y = 350)
    
    OriginalImageLabel.place(x = 210, y = 100)
    OriginalText.place(x = 290, y = 330)
    
    ResizeImageLabel.place(x = 655, y = 100)
    ResizeText.place(x = 710, y = 330)
    
    PostProccessedImage.place(x = 210,y = 350)
    ProcessedText.place(x = 245, y = 580)
    
    ActualLabel.place(x = 655, y = 400)
    CNNLabel.place(x = 655, y = 430)
    KNNLabel.place(x = 655, y = 460)
    SVMLabel.place(x = 655, y = 490)
    RFLabel.place(x = 655, y = 520)
    Backbtn.place(x = 500, y = 600)

    ImageDetailsWin.bind('<Return>',lambda event:callback())
    ImageDetailsWin.mainloop()


# ### Guessing Game

# In[8]:


#Function builds the Guessing Game Window

def OpenGuessingGame():
        
# Building the Window
    root = tk.Tk()
    root.title("Guessing Game") 
    root.geometry(str(Width)+'x'+str(Height))  
    
 # Variables
    options = ["Black Sea Sprat", "Gilt-Head Bream", "Hourse Mackerel", "Red Mullet", "Red Sea Bream", "Sea Bass", "Shrimp","Striped Red Mullet", "Trout"]
    ImagePath = os.path.join(os.getcwd(),'archive.zip','Fish_Dataset','Fish_Dataset')
    myFont = font.Font(size=15, weight='bold')
    FishGuess = tk.StringVar(root,"")
    FishGuess.set("Choose a fish")
    ActualFishLabel = tk.StringVar(root)
    ActualFishLabel.set("")
    ModelFishPrediction = tk.StringVar(root)
    ModelFishPrediction.set("")

    
# FUNCTIONS
    def GetRandomImage():
        ImageNumber = random.randint(1, 9000)
        fishfolder = options[math.floor((ImageNumber-1)/1000)]
        Img_num = ImageNumber - math.floor((ImageNumber-1)/1000)*1000
        if Img_num < 10:
            Img_num = '0000'+str(Img_num)+'.png'
        elif Img_num < 100:
            Img_num = '000'+str(Img_num)+'.png'
        elif Img_num < 1000:
            Img_num = '00'+str(Img_num)+'.png'      
        else:
            Img_num = '0'+str(Img_num)+'.png'   
        UpdatedImagePath = os.path.join(ImagePath,fishfolder,fishfolder,Img_num)
        ActualFishLabel.set(fishfolder)
        img_ = []
        with zipfile.ZipFile(os.path.join(os.getcwd(),'archive.zip'), 'r') as zipref:
            for imagepath in zipref.namelist():
                if (imagepath.__contains__('GT')):
                    pass
                elif (imagepath.__contains__('NA_Fish_Dataset')):
                    pass
                elif (imagepath.__contains__(str("/"+fishfolder+"/"+fishfolder+"/"+Img_num))): #Text was img
                    img_.append(Image.open(zipref.open(imagepath)))
        Fish_Image = img_[0]
        GetModelResults(np.array(Fish_Image))
        Fish_Image = np.array(Fish_Image).astype("uint8")
        Fish_Image = ImageTk.PhotoImage(image=Image.fromarray(Fish_Image))
        imageLabel.configure(image = Fish_Image)
        imageLabel.image = Fish_Image 


    def ValidateGuess():
        Guess = FishGuess.get()
        if str(Guess) == ActualFishLabel.get():
            IsGuessRight = "You're Right! The model predicted " + str(ModelFishPrediction.get())+"."
        else:
            IsGuessRight = "Incorrect. The actual answer is "+ActualFishLabel.get() +". The model predicted "+str(ModelFishPrediction.get())
        label.config( text = IsGuessRight )
        
    def GetModelResults(img):
        #print(load_model('CNNModel.h5').predict(cv2.resize(img,(50,50))))
        prediction = load_model('CNNModel.h5').predict(np.array([np.array(cv2.resize(img,dsize = (75,75)))]))
        CNNyhat = argmax(prediction, axis=-1).astype('int')
        ModelFishPrediction.set(options[CNNyhat[0]])
        
        


#On Screen Elements
    Titlelb = tk.Label(root, text="Guessing Game", font=('Century 20 bold'))
    GetImageButton = tk.Button(root, 
                        text="Get Image", 
                        command= GetRandomImage,
                        width=15, 
                        height=1, 
                        bg='#0052cc', 
                        fg='#ffffff')
    GuessButton = tk.Button( root,
                        text = "Guess",
                        command = ValidateGuess,
                        width=15, 
                        height=1, 
                        bg='#0052cc', 
                        fg='#ffffff'
                         )
    BackButton = tk.Button(root, 
                        text="Back", 
                        command= lambda:CloseAndOpen(root,"Main Menu"),
                        #font = myFont,
                        width=15, 
                        height=1, 
                        bg='#0052cc', 
                        fg='#ffffff')
    drop = tk.OptionMenu( root , FishGuess , *options )
    label = tk.Label( root , text = " " )
    Resultslabel = tk.Label( root , text = "" )
    imageLabel = tk.Label(root, image ="")

    GetRandomImage()  
   
 #Placing Widgets
    Titlelb.pack()
    GetImageButton.pack()
    imageLabel.pack()
    drop.pack()
    GuessButton.pack()
    label.pack()
    Resultslabel.pack()
    BackButton.pack()
    
    root.bind('<Return>',lambda event:callback())
    root.mainloop()


# ### User Guide

# In[9]:


# Function Builds the User Guide Window
def OpenUserGuide():
    # Building the Window
    root = tk.Tk()
    root.title("Guessing Game") 
    root.geometry(str(Width)+'x'+str(Height))  
    textbox = """    This app has three sections-model results, image details, and guessing game.
    
    The model results section summarizes the results of the four different models used to predict the type of seafood in each image. The models are a convolutional nueral network (CNN), K-nearest neighbors (KNN), support vector machine (SVM), and random forest (RF). You are can see each model's test confusion matrix, an ROC-AUC plot, and a metrics summary table.
    
    The image detail section allows the user to see the steps an image goes through during the modeling process. The user can search an image and see different images in the pre-processing step along with model's predictions. The first pre-processing image changes the size and clarity of the image. The second preprocessing image classifies each pixel into one of nine pixel colors. From there some of the colors are reclassified as 'Black' so that the only colors remaining are fish colors. This isolates the fish from the background.
    
    The guessing game section is a simple guessing game. An random image is selected and you need to guess what you think it is. You select the seafood type using the dropdown box and click the 'Guess' button. Text will appear letting you know if you were right and letting you know how the model would have guessed. To get a new image click the 'Get Image' button. """
    label = tk.Text(root)
    BackButton = tk.Button(root, 
                    text="Back", 
                    command= lambda:CloseAndOpen(root,"Main Menu"),
                    #font = myFont,
                    width=15, 
                    height=1, 
                    bg='#0052cc', 
                    fg='#ffffff')
   
    label.insert(tk.END, textbox)
    label.place(relx = .5, rely = .4, anchor = tk.CENTER)
    BackButton.place(relx = .5, rely = .8, anchor = tk.CENTER)
    root.bind('<Return>',lambda event:callback())
    root.mainloop()


# In[10]:


OpenMainMenu()


# In[ ]:





# In[ ]:




