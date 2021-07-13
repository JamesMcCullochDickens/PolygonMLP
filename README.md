# PolygonMLP

This repository contains the code for predicting polygon side length and area using a CNN architecture and an MLP. 

- CNNPolyPredictor.py is the CNN

- GenericDataloader.py is the pytorch dataloader that uses repeated iterators with skipping (See code)

- MLPPolyPredictor.py is the MLP 

- PolygonFactory.py generates the training and testing images, which are polygons of random radius, random number of sides from, and of a certain resolution

- The testing images are of resolutions 300/600 

- TrainAndTest.py is the training and testing code for both models

Here is a sample polygon of image resolution 300 x 300

![image](https://user-images.githubusercontent.com/23439776/125388360-5ca21a80-e36d-11eb-9d4c-1ee62b7ec6e3.png)

The best results so far are:

600 x 600:
The model accuracy for predicting the number of sides is 0.99

The root mean-squared error for area predictions is 10101.08

MLP
The model accuracy for predicting the number of sides is 0.69

The root mean-squared error for area predictions is 24605.5


300 x 300
The model accuracy for predicting the number of sides is 0.9

The root mean-squared error for area predictions is 4460.8

The model accuracy for predicting the number of sides is 0.99

The root mean-squared error for area predictions is 4118.26
