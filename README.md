# Capstone

The capstone can be split into two main files "CapStone.ipynb" or "Capstone.py" and "CapStoneUI.ipynb" or "CapstoneUI.py". 

The "CapStone" file has builds, trains, and analyzes the models. It also pulls the image dataset down from kaggle and creates the other file objects used in the "CapStoneUI" file. Running the "CapStone" file takes about 30-45 minutes, but it isn't necessary to run the "CapStoneUI" file if the image dataset is pulled down into the working directory 

There is a funciton in the CapStone file called "GetData()" which can be run to download the images into your WD if you don't want to run the whole file but need to get the images. This will ensure the folder name is the same and the code that references that folder name gets executed properly.

