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
<img src="https://user-images.githubusercontent.com/4435648/214493679-4f2f9364-e0bd-4336-a685-c1912c6d1d58.jpg" width=50% height=50%>
<br>
*Each added branch to the model produces a potential prediction that can be chosen as the accepted result.* <br>

<img src="https://user-images.githubusercontent.com/4435648/214499532-3ec1e561-47e2-48dc-be09-fceeac5cd9ab.png" width=50% height=50%> <br>
*Brevis Net reduces the average processing cost of predictions across a range of classification DNN models.*<br>


This repository contains code to build and run early exit models in tensorflow 2 along with our novel contributions of loss function, model uncertanity measurement and exit thresholding.

# How to use this repository?
[/brevis](https://github.com/SanityLacking/BrevisNet/tree/main/brevis) contains all the nessecary code to build and run the early exiting models. <br>
[/notebooks](https://github.com/SanityLacking/BrevisNet/tree/main/notebooks) contains examples of building and evaluating the early exiting models on a vareity of different DNN model types. <br>

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
Open [examplebranching.ipynb](https://github.com/SanityLacking/BrevisNet/blob/main/notebooks/Example_branching.ipynb) for a walk through of how the module is used. <br>
[notebooks/experiments](https://github.com/SanityLacking/BrevisNet/tree/main/notebooks/experiments) contains notebooks to branch and evaluate each of the tested models from the journal experiment. <br>

# Model Building
Pre-trained models can be built using scripts in [/Brevis/Raw_Models/](https://github.com/SanityLacking/BrevisNet/tree/main/brevis/raw_models). Each model was trained on Cifar10 for a minimum of 50 epochs until convergence. 

# Logging

This project uses [neptuneAI](https://www.google.com) for logging of training data, this is completely optional and only active if the module is installed.
to enable it, 
```
pip install neptune-client
pip install neptune-tensorflow-keras
```
and add your project name and credentials to neptuneCredentials.py


# Recognitions
Special thanks to [BranchyNet](https://github.com/kunglab/branchynet) who originally proposed the idea of branching models, and whose work this repo is inspired by. <br>
Dirichlet Uncertanity loss functions inspired by works from [Andrey Malinin](https://github.com/KaosEngineer/PriorNetworks)<br>
Energy based loss functions inspired by works from  [Sven Elflein](https://github.com/selflein/MA-EBM) and [Will Grathwohl](https://github.com/wgrathwohl/JEM)<br>

