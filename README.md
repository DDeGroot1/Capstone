# Capstone
Export the contents of this repository and put them in your python working directory.


## Obtaining Dataset
The first thing you will need to do is pull down the Kaggle Dataset. Run "PullDownImages.ipynb" in a Jupyter Notebook or "PullDownImages.py" in a different IDE.
This program will pull the Kaggle dataset down as a zip folder and place it in your Python working directory. The zip folder is called "archive".
This should take about 5 minutes.

If this fails you can download the dataset using the link below. Make sure to put it into your working directory and rename the downloaded zip file to "archive". Don't unzip the file.

https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset

## GUI
The GUI program is an interactive program that allows the you see the results of the models, see a few of the steps in the data pre-processing phase, and play a classification guessing game against the convolutional neural network. Run "CapstoneUI.ipynb" in a Jupyter Notebook or "CapstoneUI.py" in a different IDE. After running the program in your Python IDE a seperate window will open with the UI.

Prerequisites:
  * Dataset is pulled down from Kaggle and is located in the working directory
  * Save the "NonCNNModels" and "CNNModel" folders in the working directory. These folders contain the following models, and they are generated from the last time I ran the modeling program. 
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
The modeling program is where the images are pre-processed, the models are trained, and the models are evaluated. The models are saved in the "NonCNNModels" and "CNNModel" folders. The result plots are saved and put into the "GUI Images" folder. This program can pull down the dataset if it is not pulled down prior to running the program. Run "Capstone.ipynb" in a Jupyter Notebook or "Capstone.py" in a different IDE.

Prerequisites:
  * None
 
## Data Exploration
Since the data exploration is more of an ad hoc process that is used for feature discovery, the analysis is done in a program separate from the Modeling program. Run "CapstoneDataExploration.ipynb" in a Jupyter Notebook or "CapstoneDataExploration.py" in a different IDE. This program looks at different methods of analyzing images and builds reports, as a proof of concept, for independant consumption. These reports are saved in the "Data Exploration Reports" folder. 

Prerequisites:
  * None





