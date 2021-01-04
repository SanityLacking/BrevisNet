###evaluate the completed models. ####
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import itertools

import sys

import math
import pydot
import os
from utils import *




if __name__ == "__main__":
    opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
    args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    
    print("model to eval: {}".format(args[0]))

    #which model to eval
    #either list a valid model name, aka mnist/alexnet, etc
    #or specify a model filepath

    #which dataset to eval on?
    #check the model name for one of the valid model types and use the default dataset for that.
    

    #load the model

    #load the dataset

    #print the model structure summary

    #eval the model

    #print the results

    pass