# Capstone

## Obtaining Dataset
The first thing you will need to do is pull down the Kaggle Dataset. Run "PullDownImages.ipynb" in a Jupyter Notebook or "PullDownImages.py" in a different IDE.
This program will pull the Kaggle dataset down as a zip folder and place it in your Python working directory. The zip folder is called "archive".
This should take about 5 minutes.

## GUI
The GUI program is an interactive program that allows the you see the results of the models, see a few of the steps in the data pre-processing phase, and play a classification guessing game against the convolutional neural network. 

Prerequisites:
  * Dataset is pulled down from Kaggle and is located in the working directory
  * Save the "Models" folder in the working directory. This folder contains the following models, and they are generated from the last time I ran the modeling program. 
     - CNNModel.h5
     - KNNModel.joblib
     - SVMModel.joblib
     - RFModel.joblib
     - PixelClassifierFitted.joblib
  * Save the "GUI Images" folder in the working directory. This folder contains the following images, and they are generated from the last time I ran the modeling program.
     - ConfusionMatrix_CNN.png
     - ConfusionMatrix_KNN.png
     - ConfusionMatrix_RF.png
     - ConfusionMatrix_SVM.png
     - CountsTable.png
     - LabelTable.png
     - Roc-AUC.png
     - TimeTable.png

## Modeling



