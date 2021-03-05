import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import time

# tf.debugging.experimental.enable_dump_debug_info(logdir, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

CLASS_NAMES= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

validation_images, validation_labels = train_images[:5000], train_labels[:5000]

train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))

plt.figure(figsize=(20,20))
for i, (image, label) in enumerate(train_ds.take(5)):
    ax = plt.subplot(5,5,i+1)
    plt.imshow(image)
    plt.title(CLASS_NAMES[label.numpy()[0]])
    plt.axis('off')

def augment_images(image, label):
    # Normalize images to have a mean of 0 and standard deviation of 1
    # image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 277x277
    image = tf.image.resize(image, (227,227))
    return image, label
def augment_images2(image):
    # Normalize images to have a mean of 0 and standard deviation of 1
    # image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 277x277
    image = tf.image.resize(image, (227,227))
    return image

train_ds_size = len(list(train_ds))
train_ds_size = len(list(test_ds))
validation_ds_size = len(list(validation_ds))

train_ds = (train_ds
                  .map(augment_images)
                  .shuffle(buffer_size=train_ds_size)
                  .batch(batch_size=32, drop_remainder=True))

test_ds = (test_ds
                  .map(augment_images)
                #   .shuffle(buffer_size=train_ds_size)
                  .batch(batch_size=32, drop_remainder=True))

validation_ds = (validation_ds
                  .map(augment_images)
                #   .shuffle(buffer_size=validation_ds_size)
                  .batch(batch_size=32, drop_remainder=True))


# model = keras.models.Sequential([
#     keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)),
#     keras.layers.BatchNormalization(),
#     keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
#     keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
#     keras.layers.BatchNormalization(),
#     keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
#     keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
#     keras.layers.BatchNormalization(),
#     keras.layers.Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
#     keras.layers.BatchNormalization(),
#     keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
#     keras.layers.BatchNormalization(),
#     keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
#     keras.layers.Flatten(),
#     keras.layers.Dense(4096, activation='relu'),
#     keras.layers.Dropout(0.5),
#     keras.layers.Dense(4096, activation='relu'),
#     keras.layers.Dropout(0.5),
#     keras.layers.Dense(10, activation='softmax')
# ])


root_logdir = os.path.join(os.curdir, "logs\\fit\\")

def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)


# Keras Class API of AlexNet

class CustomAlexNet(keras.Model):
    def __init__(self):
        super(CustomAlexNet, self).__init__()
        self.conv1 = keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3))
        self.conv2 = keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same")
        self.conv3 = keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same")
        self.conv4 = keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same" )
        self.conv5 = keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same")
        self.bn = keras.layers.BatchNormalization()
        self.maxpool = keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2))
        self.dropout = keras.layers.Dropout(0.5)
        self.flatten_layer = keras.layers.Flatten()
        self.dense_layer = keras.layers.Dense(units=4096, activation='relu')
        self.output_layer = keras.layers.Dense(units=10, activation='softmax')
    
    def call(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.maxpool(x)
        x = self.dense_layer(x)
        x = self.dropout(x)
        x = self.dense_layer(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x

def buildandcompileModel(model):
    checkpoint = keras.callbacks.ModelCheckpoint("models/alexNetv4_new.hdf5", monitor='val_loss',verbose=1,save_best_only=True, mode='auto',period=1)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001,momentum=0.9), metrics=['accuracy'])
    model.summary()


    model.fit(train_ds,
          epochs=45,
          validation_data=validation_ds,
          validation_freq=1,
          callbacks=[tensorboard_cb,checkpoint])


    model.evaluate(test_ds)
    
    return model 



if __name__ == '__main__':

    inputs = keras.Input(shape=(227,227,3))
    x = keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3))(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2))(x)
    x = keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2))(x)
    x = keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(4096, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(4096, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(10, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=x, name="alexnet")
    model.save("alexnet_func_noStand.hdf5")
    
    # model = keras.models.Sequential([
    #     keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    #     keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    #     keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    #     keras.layers.Flatten(),
    #     keras.layers.Dense(4096, activation='relu'),
    #     keras.layers.Dropout(0.5),
    #     keras.layers.Dense(4096, activation='relu'),
    #     keras.layers.Dropout(0.5),
    #     keras.layers.Dense(10, activation='softmax')
    # ])

    checkpoint = keras.callbacks.ModelCheckpoint("models/alexNetv5.hdf5", monitor='val_loss',verbose=1,save_best_only=True, mode='auto',period=1)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001,momentum=0.9), metrics=['accuracy'])
    model.summary()


    model.fit(train_ds,
          epochs=50,
          validation_data=validation_ds,
          validation_freq=1,
          callbacks=[tensorboard_cb,checkpoint])


    model.evaluate(test_ds)




    # x = tf.keras.models.load_model("models/saved-model-alexnet-03-0.80.hdf5")
    # x.summary()
    # train_generator, validation_generator = loadData()
    # runAndTrainModel()
    
    # print(np.array(next(train_generator)).shape)
    
    #loadAndEvalModel()
    print("Task Complete")

    