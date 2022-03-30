#!/usr/bin/env python
# coding: utf-8

# ### Installing Required Packages

# In[1]:


pip install tk Pillow pandas numpy python-math opencv-contrib-python random2 joblib keras-models Pillow zipfile36 pathlib2 matplotlib


# ### Importing required packages

# In[2]:


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
import dataframe_image as dfi
from collections import Counter
from keras.models import load_model
from PIL import ImageTk, Image
import zipfile
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ### Setting Variables and functions

# In[3]:


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

def GetFlattenedImageForClassification(Img, w, h):
    return np.reshape(Img,(w*h,3))

def GetPixelClassifiedImage (Img,w,h,Fitted,ColorType):
    Image = np.reshape(Img,(w*h,3))
    ColorLabels = ['Red','Orange','Brown','Tan','Blue', 'LightBlue','Gray', 'White','Black']
    ColorValuesMap = [[200,75,75], [200,125,75], [151,122,83], [217,171,118],[20,20,20],[20,20,20],[20,20,20],[235,235,235],[20,20,20]]
    PredictLabels = Fitted.predict(Image)
    ImgW = PredictLabels.shape
    Color = []
    for p in range(len(PredictLabels)):
        Color.append(ColorType[(ColorLabels.index(PredictLabels[p]))])
    return np.reshape(Color,(w,h,3))

def GetTotalCounts(Image_,FittedModel):
    TotalCounts = []
    ColorLabels = ['Red','Orange','Brown','Tan','Blue', 'LightBlue','Gray', 'White','Black']
    ColorValues = [[200,75,75], [200,125,75], [151,122,83], [217,171,118],[75,75,200],[165,165,165],[125,125,125],[235,235,235],[20,20,20]]
    ColorValuesMap = [[200,75,75], [200,125,75], [151,122,83], [217,171,118],[20,20,20],[20,20,20],[20,20,20],[235,235,235],[20,20,20]]
    
    # Resizing and flattening images
    Image = GetFlattenedImageForClassification(Image_,50,50)
    PredictedLabels = Fitted.predict(Image)
    counter = Counter(PredictedLabels) 
    result = [(key, counter[key]) for key in counter]
    zeros = np.zeros(9, dtype=int)
    for i in result:
        j = ColorLabels.index(i[0])
        zeros[j] = i[1]
    TotalCounts.append(zeros)
    TotalCounts = pd.DataFrame(TotalCounts)
    ClassifiedImage = GetPixelClassifiedImage(Image_,50,50,FittedModel,ColorValues)
    ModifiedImage = GetPixelClassifiedImage(Image_,50,50,FittedModel,ColorValuesMap)
    
    TotalCounts.columns = ["Reds", "Oranges", "Browns", "Tans", "Blues", "LightBlues", "Grays", "Whites", "Blacks"]
    return ModifiedImage, ClassifiedImage, TotalCounts[["Reds", "Oranges", "Browns", "Tans", "Whites"]]


# ### Main Menu

# In[4]:


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

# In[5]:


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
            CF_img = "GUI Images\ConfusionMatrix_CNN.png"
        elif ModelSelected.get() == "K-Nearest Neighbors (KNN)":
            CF_img = "GUI Images\ConfusionMatrix_KNN.png"
        elif ModelSelected.get() == "Support Vector Machine (SVM)":
            CF_img = "GUI Images\ConfusionMatrix_SVM.png"
        else:
            CF_img = "GUI Images\ConfusionMatrix_RF.png"
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
    
    DataTable = tk.Label(ResultsWin,image = "", relief="groove")
    DataTableImage = ImageTk.PhotoImage(file = "GUI Images\TimeTable.png")
    DataTable.configure(image = DataTableImage)
    DataTable.image = DataTableImage 
    
    ROC = tk.Label(ResultsWin,image = "")    
    ROCImage = ImageTk.PhotoImage(file = "GUI Images\ROC-AUC.png")
    ROC.configure(image = ROCImage)
    ROC.image = ROCImage     
    SwapConfusionMatrix() #Gets the Initial Confusion Matrix 

    
    #Placing widgets
    drop.pack()
    GetCMButton.pack()
    CMLabel.place(relx=.3, rely = .35, anchor=tk.CENTER)
    ROC.place(relx=.7, rely = .3, anchor=tk.CENTER)
    DataTable.place(relx = .5, rely = .7, anchor = tk.CENTER)
    Backbtn.place(relx=0.5, rely=0.85, anchor=tk.CENTER)

    ResultsWin.bind('<Return>',lambda event:callback())
    ResultsWin.mainloop()


# ### Image Details

# In[6]:


# Image Detail Function creates the Image Detail Window
Fitted = joblib.load("NonCNNModels\PixelClassifierFitted.joblib")
ColorValues = [[200,75,75], [200,125,75], [151,122,83], [217,171,118],[75,75,200],[165,165,165],[125,125,125],[235,235,235],[20,20,20]]
ColorValuesMap = [[200,75,75], [200,125,75], [151,122,83], [217,171,118],[20,20,20],[20,20,20],[20,20,20],[235,235,235],[20,20,20]]


def OpenImageDetails():
    #Create an instance of Tkinter frame or window
    ImageDetailsWin= tk.Tk()
    ImageDetailsWin.title("Image Details")    
    ImageDetailsWin.geometry(str(Width)+'x'+str(Height))
    
# Variables       
    myFont = font.Font(size=15, weight='bold')
    options = ["Black Sea Sprat", "Gilt-Head Bream", "Hourse Mackerel", "Red Mullet", "Red Sea Bream", "Sea Bass", "Shrimp","Striped Red Mullet", "Trout"]
    ImagePath = os.path.join(os.getcwd(),'archive.zip','Fish_Dataset','Fish_Dataset')  

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
                
                # Adding Blur and Resizing Image
                BlurAndResizedImageList = []
                BlurImage = np.array(Fish_Image)
                BlurImage = cv2.blur(BlurImage,(random.randint(1,3),random.randint(1,3)))
                BlurAndResizedImageList.append(cv2.resize(BlurImage,(50,50)))
                BlurAndResizedImage = np.array(BlurAndResizedImageList)
                
                #Get Model Labels
                CNNprediction = load_model('CNNModel\CNNModel.h5').predict(BlurAndResizedImage)
                CNNyhat = argmax(CNNprediction, axis=-1).astype('int')
                pred = options[CNNyhat[0]]

                ResizeImage = BlurAndResizedImage
                ModifiedImage, ReclassifiedImage, counts = GetTotalCounts(ResizeImage, Fitted)
                
                CountsTable = pd.DataFrame(counts)
                dfi.export(CountsTable,"GUI Images\CountsTable.png")
                CountsTable_Png = ImageTk.PhotoImage(file = "GUI Images\CountsTable.png")

                KNNpredict = joblib.load('NonCNNModels\KNNModel.joblib').predict(counts)
                SVMpredict = joblib.load('NonCNNModels\SVMModel.joblib').predict(counts)
                RFpredict = joblib.load('NonCNNModels\RFModel.joblib').predict(counts)
 
                LabelTable = np.array([[fishfolder, pred,options[KNNpredict[0]], options[SVMpredict[0]],options[RFpredict[0]]]])
                LabelTable = pd.DataFrame(LabelTable)
                LabelTable.columns = ["Actual Label","CNN Prediction", "KNN Prediction", "SVM Prediction", "Random Forest Prediction"]
                dfi.export(LabelTable,"GUI Images\LabelTable.png")                
                LabelTablePng = ImageTk.PhotoImage(file = "GUI Images\LabelTable.png")
                
                Fish_Image = Resize(np.array(Fish_Image), 200, 200)
                ResizeImage = Resize(BlurAndResizedImage[0], 200,200)
                ReclassifiedImage = np.array(ReclassifiedImage, dtype ='uint8')
                ReclassifiedImage = Resize(ReclassifiedImage,200,200)
                ModifiedImage = np.array(ModifiedImage, dtype='uint8')
                ModifiedImage = Resize(ModifiedImage,200,200)

                
                Fish_Image = np.array(Fish_Image).astype("uint8")
                Fish_Image = ImageTk.PhotoImage(image=Image.fromarray(Fish_Image))
                ResizeImage = np.array(ResizeImage).astype("uint8")
                ResizeImage = ImageTk.PhotoImage(image=Image.fromarray(ResizeImage))
                ReclassifiedImage =np.array(ReclassifiedImage).astype("uint8")
                ReclassifiedImage = ImageTk.PhotoImage(image=Image.fromarray(ReclassifiedImage))
                ModifiedImage = np.array(ModifiedImage).astype("uint8")
                ModifiedImage = ImageTk.PhotoImage(image=Image.fromarray(ModifiedImage))
            
                
                OriginalImageLabel.configure(image = Fish_Image)
                OriginalImageLabel.image = Fish_Image 
                ResizeImageLabel.configure(image = ResizeImage)
                ResizeImageLabel.image = ResizeImage 
                ReclassifiedImageLabel.configure(image = ReclassifiedImage)
                ReclassifiedImageLabel.image = ReclassifiedImage
                PostProccessedImage.configure(image = ModifiedImage)
                PostProccessedImage.image = ModifiedImage 
                CountsTable_PngLabel.configure(image = CountsTable_Png)
                CountsTable_PngLabel.image = CountsTable_Png
                LabelTablePngLabel.configure(image = LabelTablePng)
                LabelTablePngLabel.image = LabelTablePng

                
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
    
    OriginalImageLabel = tk.Label(ImageDetailsWin, image ="", relief="groove")
    ResizeImageLabel = tk.Label(ImageDetailsWin, image ="", relief="groove")
    ReclassifiedImageLabel = tk.Label(ImageDetailsWin, image ="", relief="groove")
    PostProccessedImage = tk.Label(ImageDetailsWin, image ="" , relief="groove")
    
    CountsTable_PngLabel = tk.Label(ImageDetailsWin, image ="" , relief="groove")
    LabelTablePngLabel = tk.Label(ImageDetailsWin, image = "" , relief="groove")
    
    OriginalText = tk.Label(ImageDetailsWin, text = "Original Image")
    ResizeText = tk.Label(ImageDetailsWin, text = "Resized (50 x 50)")    
    ReclassifiedImageText = tk.Label(ImageDetailsWin, text = "Classified Pixel Colors")
    ProcessedText = tk.Label(ImageDetailsWin, text = "Turn Some Colors to Black")
    
    TextBox = tk.Text(ImageDetailsWin, wrap = tk.WORD)
    textbox = """    This screen demonstrates how the images are prepared before training the models. The first image displays the original image. The image is then compressed into a 50 x 50 pixel image with a bit of random blur added. This removes the noise and small details within the image. At this point the images are ready for the CNN algorithm, but more work is needed for the KNN, SVM, and Random Forest algorithms.
    The next step reduces the variance between each pixel color. Each pixel is classified as one of nine different colors based on what the color it is most alike. The nine colors are Red, Orange, Brown, Tan, Blue, LightBlue, Gray, White, and Black. The third image shows this transformation.
    Next, the blue, lightblue, and gray pixels are turned into black. This removes most of the background noise and isolates the colors associated with the seafood from the black background. This is seen in the forth image.
    The final step counts the number of pixels of each color type of the image. The totals per image are the features in the KNN, SVM, and Random Forest models. The totals can be seen in the table to the right."""
   
    GetImage(1)
    TextBox.insert(tk.END, textbox)      
    # Placing Widgets
   
    LabelTablePngLabel.place(x = 185, y = 25) 
      
    OriginalImageLabel.place(x = 56, y = 115)
    OriginalText.place(x = 112, y = 320)
    
    ResizeImageLabel.place(x = 312, y = 115)
    ResizeText.place(x = 367, y = 320)
    
    ReclassifiedImageLabel.place(x = 568, y = 115)
    ReclassifiedImageText.place(x = 610, y = 320)
    
    PostProccessedImage.place(x = 824,y = 115)
    ProcessedText.place(x = 852, y = 320)
    
    CountsTable_PngLabel.place(x = 710, y = 370)
    
    GetImageButton.place(x = 878, y = 440)
    ImageEntry.place(x = 745, y = 440)
    ErrorLabel.place(x = 750, y = 475)

    Backbtn.place(x = 800, y = 600)
    TextBox.place(x = 25, y = 370, height = 270, width = 630)

    ImageDetailsWin.bind('<Return>',lambda event:callback())
    ImageDetailsWin.mainloop()


# ### Guessing Game

# In[7]:


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
            IsGuessRight = "Incorrect. The actual answer is "+ActualFishLabel.get() +". The model predicted "+str(ModelFishPrediction.get()+".")
        label.config( text = IsGuessRight )
        
    def GetModelResults(img):
        #print(load_model('CNNModel.h5').predict(cv2.resize(img,(50,50))))
        prediction = load_model('CNNModel\CNNModel.h5').predict(np.array([np.array(cv2.resize(img,dsize = (50,50)))]))
        CNNyhat = argmax(prediction, axis=-1).astype('int')
        ModelFishPrediction.set(options[CNNyhat[0]])
        
        


#On Screen Elements
    Titlelb = tk.Label(root, text="Guessing Game", font=('Century 20 bold'))
    GetImageButton = tk.Button(root, 
                        text="Get New Image", 
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
    imageLabel = tk.Label(root, image ="", relief="groove")

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

# In[8]:


# Function Builds the User Guide Window
def OpenUserGuide():
    # Building the Window
    root = tk.Tk()
    root.title("Guessing Game") 
    root.geometry(str(Width)+'x'+str(Height))  
    textbox = """    This app has three sections-model results, image details, and guessing game.
    
    The model results section summarizes the results of the four different models used to predict the type of seafood in each image. The models are a convolutional nueral network (CNN), K-nearest neighbors (KNN), support vector machine (SVM), and random forest (RF). You are can see each model's test confusion matrix, an ROC-AUC plot, and a metrics summary table.
    
    The image detail section allows the user to see the steps an image goes through during the modeling process. The user can search an image and see different snapshots in the pre-processing phase along with model's predictions. The first pre-processing image changes the size and clarity of the image. The second preprocessing image classifies each pixel into one of nine pixel colors. From there some of the colors are reclassified as 'Black' so that the only colors remaining are fish colors. This isolates the fish from the background.
    
    The guessing game section is a simple guessing game. An random image is selected and you need to guess what you think it is. You select the seafood type using the dropdown box and click the 'Guess' button. Text will appear letting you know if you were right and letting you know how the model would have guessed. To get a new image click the 'Get Image' button. """
    label = tk.Text(root, wrap=tk.WORD)
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


# In[9]:


OpenMainMenu()


# In[ ]:





# In[ ]:





# In[ ]:




