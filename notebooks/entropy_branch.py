import tensorflow as tf
import numpy as np
import sys
sys.path.append("..") # Adds higher directory to python modules path.
# import branchingdnn as branching
# from branchingdnn.utils import * 

# Download MNIST dataset
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
# print(y_train)
K= 10 # number of classes

train_labels = tf.keras.utils.to_categorical(train_labels,10)
test_labels = tf.keras.utils.to_categorical(test_labels,10)

validation_size = 5000
shuffle_size = 22500
batch_size=32
validation_images, validation_labels = train_images[:validation_size], train_labels[:validation_size] #get the first 5k training samples as validation set
train_images, train_labels = train_images[validation_size:], train_labels[validation_size:] # now remove the validation set from the training set.
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))


def augment_images(image, label,input_size=(227,227), channel_first = False):
            # Normalize images to have a mean of 0 and standard deviation of 1
            # image = tf.image.per_image_standardization(image)
            # Resize images from 32x32 to 277x277
            image = tf.image.resize(image,input_size)
            if channel_first:
                image = tf.transpose(image, [2, 0, 1])
            
            return image, label

train_ds_size = len(list(train_ds))
test_ds_size = len(list(test_ds))
validation_ds_size = len(list(validation_ds))
train_ds = (train_ds.map(augment_images))
validation_ds = (validation_ds.map(augment_images))
test_ds = (test_ds.map(augment_images))


train_ds = (train_ds.map(augment_images))
validation_ds = (validation_ds.map(augment_images))
test_ds = (test_ds.map(augment_images))

target = tf.data.Dataset.from_tensor_slices((train_labels))
train_ds = tf.data.Dataset.zip((train_ds,target))

v_target = tf.data.Dataset.from_tensor_slices((validation_labels))
validation_ds = tf.data.Dataset.zip((validation_ds,v_target))

t_target = tf.data.Dataset.from_tensor_slices((test_labels))
test_ds = tf.data.Dataset.zip((test_ds,t_target))


print("trainSize {}".format(train_ds_size))
print("testSize {}".format(test_ds_size))
train_ds = (train_ds
                
                .shuffle(buffer_size=tf.cast(shuffle_size,'int64'))
                .batch(batch_size=batch_size, drop_remainder=True))

test_ds = (test_ds
               
                #   .shuffle(buffer_size=train_ds_size)
                .batch(batch_size=1, drop_remainder=True))

validation_ds = (validation_ds
               
                #   .shuffle(buffer_size=validation_ds_size)
                .batch(batch_size=batch_size, drop_remainder=True))


def loss_function(annealing_rate=1, momentum=1, decay=1, global_loss=False):
    def crossEntropy_loss(labels, outputs): 
#         softmax = tf.nn.softmax(outputs)
        loss = tf.keras.losses.categorical_crossentropy(labels, outputs)
        return loss
    return  crossEntropy_loss

def calcEntropy_Tensors(y_hat):
        rank = tf.rank(y_hat)
        def calc_E(y_hat):
            results = tf.clip_by_value((tf.math.log(y_hat)/tf.math.log(tf.constant(2, dtype=y_hat.dtype))), -1e12, 1e12)
            return (y_hat * results)
        sumEntropies = (tf.map_fn(calc_E,tf.cast(y_hat,'float')))
        if rank == 1:
            sumEntropies = tf.reduce_sum(sumEntropies)
        return -sumEntropies

def calcEntropy_Tensors2(y_hat):
    #entropy is the sum of y * log(y) for all possible labels.
    #doesn't deal with cases of log(0)
    val = y_hat * tf.math.log(y_hat)/tf.math.log(tf.constant(2, dtype=y_hat.dtype))
    sumEntropies =  tf.reduce_mean(tf.boolean_mask(val,tf.math.is_finite(val)))
    return -sumEntropies

class EntropyEndpoint(tf.keras.layers.Layer):
        def __init__(self, num_outputs, name=None, **kwargs):
            super(EntropyEndpoint, self).__init__(name=name)
            self.num_outputs = num_outputs
            self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
        def build(self, input_shape):
            self.kernel = self.add_weight("kernel", shape=[int(input_shape[-1]), self.num_outputs])

        def get_config(self):
            config = super().get_config().copy()
            config.update({
                'num_outputs': self.num_outputs,
                'name': self.name
            })
            return config

        def call(self, inputs, labels,learning_rate=1):
            outputs = tf.matmul(inputs,self.kernel)
            outputs = tf.nn.softmax(outputs)
            entropy = calcEntropy_Tensors(outputs)
            # tf.print("entropy",tf.reduce_sum(entropy))

            # print(entropy)
            pred = tf.argmax(outputs,1)
            truth = tf.argmax(labels,1)
            match = tf.reshape(tf.cast(tf.equal(pred, truth), tf.float32),(-1,1))
            # total_entropy = tf.reduce_sum([entropy],1, keepdims=True)
            # tf.print("match",tf.reduce_sum(match+1e-20), (tf.reduce_sum(tf.abs(1-match))+1e-20) )
            # tf.print("succ",entropy*match)
            # tf.print("fail",entropy*(1-match))

            # tf.print("succ",tf.reduce_sum(entropy*match),tf.reduce_sum(match+1e-20))
            # tf.print("fail",tf.reduce_sum(entropy*(1-match)),(tf.reduce_sum(tf.abs(1-match))+1e-20) )
            mean_succ = tf.reduce_sum(entropy*match) / tf.reduce_sum(match+1e-20)
            mean_fail = tf.reduce_sum(entropy*(1-match)) / (tf.reduce_sum(tf.abs(1-match))+1e-20) 
            
            self.add_metric(entropy, name=self.name+"_entropy")
            # self.add_metric(total_entropy, name=self.name+"_entropy",aggregation='mean')
            self.add_metric(mean_succ, name=self.name+"_mean_ev_succ",aggregation='mean')
            self.add_metric(mean_fail, name=self.name+"_mean_ev_fail",aggregation='mean')
            
            return outputs


class softmax_custom(tf.keras.layers.Layer):
        def __init__(self, num_outputs, name=None, **kwargs):
            super(softmax_custom, self).__init__(name=name)
            self.num_outputs = num_outputs
            self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
        def build(self, input_shape):
            self.kernel = self.add_weight("kernel", shape=[int(input_shape[-1]), self.num_outputs])
            print(self.kernel)

        def get_config(self):
            config = super().get_config().copy()
            config.update({
                'num_outputs': self.num_outputs,
                'name': self.name
            })
            return config

        def call(self, inputs, labels,learning_rate=1):
            outputs = tf.matmul(inputs,self.kernel)
            # outputs = inputs
            # tf.print(inputs)
            outputs = tf.nn.softmax(outputs)
            entropy = calcEntropy_Tensors(outputs)
            tf.print("entropy2",tf.reduce_sum(entropy))
            pred = tf.argmax(outputs,1)
            truth = tf.argmax(labels,1)
            match = tf.reshape(tf.cast(tf.equal(pred, truth), tf.float32),(-1,1))
            mean_succ = tf.reduce_sum(entropy*match) / tf.reduce_sum(match+1e-20)
            mean_fail = tf.reduce_sum(entropy*(1-match)) / (tf.reduce_sum(tf.abs(1-match))+1e-20) 
            

            self.add_metric(entropy, name=self.name+"_entropy2")
            # self.add_metric(mean_succ, name=self.name+"_mean2_ev_succ",aggregation='mean')
            # self.add_metric(mean_fail, name=self.name+"_mean2_ev_fail",aggregation='mean')
            return outputs

outputs =[]
targets = tf.keras.Input(shape=(10,),name='targets')
inputs = tf.keras.Input(shape=(227,227,3))
x = tf.keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3))(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2))(x)

##branch one
branchLayer = tf.keras.layers.Flatten(name=tf.compat.v1.get_default_graph().unique_name("branch_flatten"))(x)
branchLayer = tf.keras.layers.Dense(124, activation="relu",name=tf.compat.v1.get_default_graph().unique_name("branch124"))(branchLayer)
branchLayer = tf.keras.layers.Dense(64, activation="relu",name=tf.compat.v1.get_default_graph().unique_name("branch64"))(branchLayer)
output = EntropyEndpoint(10, name=tf.compat.v1.get_default_graph().unique_name("branch_endpoint"))(branchLayer, targets)
outputs.append(output)



x = tf.keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2))(x)
x = tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same")(x)
x = tf.keras.layers.BatchNormalization()(x)


##branch two
branchLayer = tf.keras.layers.Flatten(name=tf.compat.v1.get_default_graph().unique_name("branch_flatten"))(x)
branchLayer = tf.keras.layers.Dense(124, activation="relu",name=tf.compat.v1.get_default_graph().unique_name("branch124"))(branchLayer)
branchLayer = tf.keras.layers.Dense(64, activation="relu",name=tf.compat.v1.get_default_graph().unique_name("branch64"))(branchLayer)
output = EntropyEndpoint(10, name=tf.compat.v1.get_default_graph().unique_name("branch_endpoint"))(branchLayer, targets)
outputs.append(output)

x = tf.keras.layers.Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2))(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(4096, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)

##branch three
branchLayer = tf.keras.layers.Flatten(name=tf.compat.v1.get_default_graph().unique_name("branch_flatten"))(x)
branchLayer = tf.keras.layers.Dense(124, activation="relu",name=tf.compat.v1.get_default_graph().unique_name("branch124"))(branchLayer)
branchLayer = tf.keras.layers.Dense(64, activation="relu",name=tf.compat.v1.get_default_graph().unique_name("branch64"))(branchLayer)
output = EntropyEndpoint(10, name=tf.compat.v1.get_default_graph().unique_name("branch_endpoint"))(branchLayer, targets)
outputs.append(output)


x = tf.keras.layers.Dense(4096, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(10, activation='softmax')(x)
outputs.insert(0,output)

# output = EntropyEndpoint(10, name=tf.compat.v1.get_default_graph().unique_name("branch_softmax"))(branchLayer, targets)
# output2 = softmax_custom(10, name=tf.compat.v1.get_default_graph().unique_name("branch_softmax"))(branchLayer_alt, targets)

model = tf.keras.Model(inputs=[inputs,targets], outputs=outputs, name="entropy_results")
loss_fn = loss_function()
model.compile( loss=loss_fn, optimizer=tf.optimizers.SGD(lr=0.001,momentum=0.9), metrics=['accuracy'])

import os
import time
root_logdir = os.path.join(os.curdir, "logs\\fit\\")
checkpoint = tf.keras.callbacks.ModelCheckpoint("alexnet_entropy_results2.hdf5", monitor='val_loss',verbose=1,save_best_only=True, mode='auto')
def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)

model.fit(train_ds,
        epochs=20,
        validation_data=validation_ds,
        validation_freq=1,
        # batch_size=1,
        verbose=1,
        callbacks=[tensorboard_cb,checkpoint])