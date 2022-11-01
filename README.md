# BrevisNet
This Repository contains the code for the Journal Paper: 
>**Resource-Adaptive and OOD-Robust Inference of Deep Neural Networks on IoT Devices** <br>
>Cailen Robertson, Thanh Tam Nguyen, Quoc Viet Hung Nguyen, Jun Jo

![branch_graph_simple](https://user-images.githubusercontent.com/4435648/184592723-668a24ab-a96e-4b07-8f40-aeb0653dad95.png)
![branch_graph_simple_2](https://user-images.githubusercontent.com/4435648/184592743-75b106f2-4803-4e89-b08d-708cf3f4c7aa.png)<br>
_Fig 1. Output predictions at a branch exit with previous(Entropy) techniques compared to our (Energy) threshold and loss techniques._

This Repository contains both the source models for building branching models efficiently in Tensorflow and python, as well as providing several Jupyter Notebooks containing implementation for the building, training, and evaluating of branching models.

# What is branching/early exiting?

Early exiting is a deep learning model augmentation technique where additional classfifier exits are added to pre-existing model. The added classifier exits and their produced predictions become potential results for a given input, and if chosen as the accepted output, mean that the rest of the model's layers do not need to be processed, saving time and energy.
<br>
This repository contains code to build and run early exit models in tensorflow 2 along with our novel contributions of loss function, model uncertanity measurement and exit thresholding.

# How to use this repository?
/brevis contains all the nessecary code to build and run the early exiting models. <br>
/[Notebooks](https://github.com/SanityLacking/BrevisNet/tree/main/notebooks) contains examples of building and evaluating the early exiting models on a vareity of different DNN model types. <br>

# Requirements
  Tensorflow 2.+ <br>
  Python 3.7 + <br>
  Jupyter <br>

# Setup

Clone the Repository 
```
git clone https://github.com/SanityLacking/BrevisNet.git
```
Access the notebooks via Jupyter
```
cd BrevisNet
cd notebooks
jupyter lab
```
Open [examplebranching.ipynb]()

# Logging

This project uses [neptuneAI](https://www.google.com) for logging of training data, this is completely optional and only active if the module is installed.
to enable it, 
```
pip install neptune-client
pip install neptune-tensorflow-keras
```
and add your project name and credentials to neptuneCredentials.py


