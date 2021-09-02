# import the necessary packages
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import itertools

# from keras.models import load_model
# from keras.utils import CustomObjectScope
# from keras.initializers import glorot_uniform

import math
import pydot
import os
#os.environ["PATH"] += os.pathsep + "C:\Program Files\Graphviz\bin"
#from tensorflow.keras.utils import plot_model
# from utils import *

# from Alexnet_kaggle_v2 import * 
import branchingdnn as branching
from branchingdnn.utils import *

# ALEXNET = False
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)
root_logdir = os.path.join(os.curdir, "logs\\fit\\")


# tf.debugging.experimental.enable_dump_debug_info("logs/", tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)




def loss_function(annealing_rate=1, momentum=1, decay=1, global_loss=False):
    def crossEntropy_loss(labels, outputs): 
#         softmax = tf.nn.softmax(outputs)
        loss = tf.keras.losses.categorical_crossentropy(labels, outputs)
        return loss
    return  crossEntropy_loss
    
class EntropyEndpoint(tf.keras.layers.Layer):
        def __init__(self, num_outputs, name=None, **kwargs):
            super(EntropyEndpoint, self).__init__(name=name)
            self.num_outputs = num_outputs
            self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
        # def build(self, input_shape):
        #     self.kernel = self.add_weight("kernel", shape=[int(input_shape[-1]), self.num_outputs])

        def get_config(self):
            config = super().get_config().copy()
            config.update({
                'num_outputs': self.num_outputs,
                'name': self.name
            })
            return config

        def call(self, inputs, labels,learning_rate=1):
            # outputs = tf.matmul(inputs,self.kernel)
            # outputs = inputs
            tf.print("inputs",inputs)
            outputs = tf.nn.softmax(inputs)
            tf.print("softmax",outputs)
            entropy = calcEntropy_Tensors(inputs)
            # tf.print("entropy",entropy)
            # print(entropy)
            pred = tf.argmax(outputs,1)
            # tf.print("pred", pred)
            truth = tf.argmax(labels,1)
            # tf.print("truth", truth)
            match = tf.reshape(tf.cast(tf.equal(pred, truth), tf.float32),(-1,1))
            # tf.print("match", match)
            # tf.print("match",match)

            # tf.print("succ",tf.reduce_sum(entropy*match),tf.reduce_sum(match+1e-20))
            # tf.print("fail",tf.reduce_sum(entropy*(1-match)),(tf.reduce_sum(tf.abs(1-match))+1e-20) )
            mean_succ = tf.reduce_sum(entropy*match) / tf.reduce_sum(match+1e-20)
            mean_fail = tf.reduce_sum(entropy*(1-match)) / (tf.reduce_sum(tf.abs(1-match))+1e-20) 
            
            self.add_metric(entropy, name=self.name+"_entropy")
            # self.add_metric(total_entropy, name=self.name+"_entropy",aggregation='mean')
            self.add_metric(mean_succ, name=self.name+"_mean_ev_succ",aggregation='mean')
            self.add_metric(mean_fail, name=self.name+"_mean_ev_fail",aggregation='mean')
            
            return inputs


if __name__ == "__main__":
    # x = branching.core.Run_alexNet( 20, modelName="alexNetv6.hdf5", saveName = "alexNetv6_compress",transfer = True ,customOptions="CrossE")
    # x = branching.models.SelfDistilation.alexnet( 20, modelName="alexNetv6.hdf5", saveName = "alexNetv6_distil_BN_only",transfer = True,customOptions="CrossE")
    
    # x = tf.keras.models.load_model('notebooks/alexnet_entropy_results.hdf5',custom_objects={"EntropyEndpoint":EntropyEndpoint,"crossEntropy_loss":loss_function()})
    x = tf.keras.models.load_model('notebooks/alexnet_entropy_results.hdf5', custom_objects={"EntropyEndpoint":EntropyEndpoint,"crossEntropy_loss":loss_function()})
    y = branching.core.GetResultsCSV(x, prepare.dataset(tf.keras.datasets.cifar10.load_data(),64,5000,22500,(227,227),include_targets=True),"entropy_results")
    y = branching.core.evalModel(x, tf.keras.datasets.cifar10.load_data(),"compressed")
  
    # x = tf.keras.models.load_model("models/alexNetv5_crossE.hdf5")
    # y = branching.core.GetResultsCSV(x, tf.keras.datasets.cifar10.load_data(),"_compressed")
  
    # x = tf.keras.models.load_model("models/alexNetv5_crossE_Eadd.hdf5")
    # y = branching.GetResultsCSV(x, tf.keras.datasets.cifar10.load_data(),"_crossE_Eadd")
    


    # x = branching.Run_alexNet( 10, modelName="alexNetv5.hdf5", saveName = "alexNetv5_customLoss_3",transfer = False, custom=True)

    


    # x = tf.keras.models.load_model("models/alexNetv5_customLoss_2.hdf5")
    # y = branching.GetResultsCSV(x, tf.keras.datasets.cifar10.load_data(),"custloss_2")

    # x = branching.Run_inceptionv3( 3, modelName="inception_finetuned.hdf5", saveName = "inception_branched",transfer = False)
    # x = branching.Run_resnet50v2( 3, modelName="resnet50_finetuned.hdf5", saveName = "resnet50_branched",transfer = False)

    # x = branching.Run_mnistNet( 5, modelName="mnistNormal.hdf5", saveName = "mnistNormal_branched",transfer = True)
    
    """
    Various model versions:
        alexNetv5 : up to date version of testing, trained using the augmented, not self standardized images. base for most other versions that I tried out
        
        models with alt in the name are models I made trying to track down what is going on with the missing 0 class from branches
        alexNetv5_alt6: model with branches on dense layers, this model actually lost a second class completely as well, class 1. 
    """

    # x = tf.keras.models.load_model("models/mnist_transfer_trained_21-01-04_125846.hdf5")
    # x.summary()
    # branching.eval_branches(x,branching.loadTrainingData(),1,"accuracy")
    # branching.eval_branches(x,branching.loadTrainingData(),1,"entropy")
    # branching.find_mistakes(x,branching.loadTrainingData(),1)
    
    # x = tf.keras.models.load_model("models/alexnet_branched_new_trained.hdf5")
    # x.summary()
    # branching.entropyMatrix(x,tf.keras.datasets.cifar10.load_data())


    # branching.eval_branches(x,tf.keras.datasets.cifar10.load_data(),1,"entropy")
    # branching.find_mistakes(x,tf.keras.datasets.cifar10.load_data(),1)



    # x.summary()
    # branching.eval_branches(x,tf.keras.datasets.cifar10.load_data(),1,)



    ####Make a new model
    # x = branching.Run_alexNet(50, saveName = "alexnext_branched_fullModel_trained",transfer = False)
    # x.summary()

    # branching.datasetStats(tf.keras.datasets.cifar10.load_data())

    # x = tf.keras.models.load_model("models/alexnext_branched_fullModel_trained_branched_branched.hdf5")
    # x.summary()
    # branching.eval_branches(x,tf.keras.datasets.cifar10.load_data())
    # """ 
    # x = tf.keras.models.load_model("models/alexnet_branch_pooling.hdf5")
    # x.summary()
    # branching.eval_branches(x,tf.keras.datasets.cifar10.load_data())
    # x = tf.keras.models.load_model("models/alexnet_branched_new_trained.hdf5")
    # x.summary()
    # branching.eval_branches(x,tf.keras.datasets.cifar10.load_data())
    # """
    # x = branching.Run_train_model(x,tf.keras.datasets.cifar10.load_data(),10)
    # x.save("models/alexnet_branched_new_trained.hdf5")

    # x = branching.Run_alexNet(1)

    # x = branching.mnistbranching()
    

    # x = branching.loadModel("models/mnist_trained_20-12-15_112434.hdf5")
    # x = tf.keras.models.load_model("models/mnist2_transfer_trained_.tf")

    # x.save("models/mnistNormal2_trained.hdf5")
    # saveModel(x,"mnist2_transfer_trained_final",includeDate=False)
    pass

