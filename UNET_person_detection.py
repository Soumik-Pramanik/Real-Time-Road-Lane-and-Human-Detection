from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x
# def encoder_block(inputs,num_filters):
#   x=conv_block(inputs,num_filters)
#   p=MaxPool2D((2,2))(x)
#   return x,p

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_resnet50_unet(input_shape):
    """ Input """
    inputs = Input(input_shape)

    """ Pre-trained ResNet50 Model """
    resnet50 = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)

    """ Encoder """
    s1 = resnet50.get_layer("input_1").output           ## (512 x 512)
    s2= resnet50.get_layer("conv1_relu").output        ## (256 x 256)
    s3 = resnet50.get_layer("conv2_block3_out").output  ## (128 x 128)
    s4 = resnet50.get_layer("conv3_block4_out").output  ## (64 x 64)

    """ Bridge """
    b1 = resnet50.get_layer("conv4_block6_out").output  ## (32 x 32)

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)                     ## (64 x 64)
    d2 = decoder_block(d1, s3, 256)                     ## (128 x 128)
    d3 = decoder_block(d2, s2, 128)                     ## (256 x 256)
    d4 = decoder_block(d3, s1, 64)                      ## (512 x 512)

    """ Output """
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="ResNet50_U-Net")
    return model

if __name__ == "__main__":
    input_shape = (512, 512, 3)
    model = build_resnet50_unet(input_shape)
    model.summary()
    
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["acc"])
model.summary()

import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from scipy.io import loadmat
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications import Xception
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import cv2
# import tensorflow_addons as tfa
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config = config)
print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))

train_path = '/content/drive/MyDrive/PennFudanPed/PNGImages'
test_path = '/content/drive/MyDrive/PennFudanPed/dataset_ped/test'
validation_path = '/content/drive/MyDrive/PennFudanPed/dataset_ped/validation'

# creating a empty list and iterating over every files in image folder and converting them into array and appending them in the created list.
train_images = []
for filename in sorted(os.listdir(train_path)):
    train_image_fullpath = os.path.join(train_path, filename)
    train_img = load_img(train_image_fullpath, target_size=(512,512)) #default input size of the vgg models = (224,224)
    train_img_arr = img_to_array(train_img)
    train_images.append(train_img_arr)

# Converting list into numpy arrays
train_images = np.array(train_images)

# creating a empty list and iterating over every files in image folder and converting them into array and appending them in the created list.
val_images = []
for filename in sorted(os.listdir(validation_path)):
    val_image_fullpath = os.path.join(validation_path, filename)
    val_img = load_img(val_image_fullpath, target_size=(512,512)) #default input size of the vgg models = (224,224)
    val_img_arr = img_to_array(val_img)
    val_images.append(val_img_arr)
  
# Converting list into numpy arrays
val_images = np.array(val_images)

# creating a empty list and iterating over every files in image folder and converting them into array and appending them in the created list.

test_images = []
for filename in sorted(os.listdir(test_path)):
    test_image_fullpath = os.path.join(test_path, filename)
    test_img = load_img(test_image_fullpath, target_size=(512,512)) #default input size of the vgg models = (224,224)
    test_img_arr = img_to_array(test_img)
    test_images.append(test_img_arr)

# Converting list into numpy arrays
test_images = np.array(test_images)

# Paths of train, test and validation bounding box coordinates
train_path1 = '/content/drive/MyDrive/PennFudanPed/mask/all_train'
test_path1 = '/content/drive/MyDrive/PennFudanPed/mask/test'
validation_path1 = '/content/drive/MyDrive/PennFudanPed/mask/validation'

# creating a empty list and iterating over every files in image folder and converting them into array and appending them in the created list.
train_target_images = []
for filename in sorted(os.listdir(train_path1)):
    train_image_fullpath = os.path.join(train_path1, filename)
    image_bro=cv2.imread(train_image_fullpath)
    image_bro=cv2.resize(image_bro,(512,512),interpolation = cv2.INTER_NEAREST)
    image_bro_gray=cv2.cvtColor(image_bro, cv2.COLOR_BGR2GRAY)
    train_target_images.append(image_bro_gray)

# Converting list into numpy arrays
train_targets = np.array(train_target_images)

train_images.shape
train_targets.shape

from google.colab.patches import cv2_imshow
image_bro=cv2.imread(train_image_fullpath)
image_bro=cv2.resize(image_bro,(512,512),interpolation = cv2.INTER_NEAREST)
image_bro_gray=cv2.cvtColor(image_bro, cv2.COLOR_BGR2GRAY)
# cv2_imshow(image_bro_gray)
type(image_bro_gray)

test_target_images = []
for filename in sorted(os.listdir(test_path1)):
    test_image_fullpath = os.path.join(test_path1, filename)
    image_bro=cv2.imread(test_image_fullpath)
    image_bro=cv2.resize(image_bro,(512,512),interpolation = cv2.INTER_NEAREST)
    image_bro_gray=cv2.cvtColor(image_bro, cv2.COLOR_BGR2GRAY)
    test_target_images.append(image_bro_gray)

# Converting list into numpy arrays
test_targets = np.array(test_target_images)

# creating a empty list and iterating over every files in image folder and converting them into array and appending them in the created list.
val_target_images = []
for filename in sorted(os.listdir(validation_path1)):
    val_image_fullpath = os.path.join(validation_path1, filename)
    image_bro=cv2.imread(val_image_fullpath)
    image_bro=cv2.resize(image_bro,(512,512),interpolation = cv2.INTER_NEAREST)
    image_bro_gray=cv2.cvtColor(image_bro, cv2.COLOR_BGR2GRAY)
    val_target_images.append(image_bro_gray)
  
# Converting list into numpy arrays
val_targets = np.array(val_target_images)

# Train the model for 20 epochs
trained_model = model.fit(train_images, train_targets,
             validation_data=(val_images, val_targets),
             batch_size=4,
             epochs=20,
             shuffle=True,
             verbose=2)

from tensorflow.keras.models import load_model
model.save('/content/drive/MyDrive/PennFudanPed/model2_resnet50.h5')
from tensorflow.keras.models import load_model
model = load_model('/content/drive/MyDrive/PennFudanPed/model_resnet50.h5')
import cv2
from google.colab.patches import cv2_imshow

# test_img=load_img('/content/drive/MyDrive/PennFudanPed/Test/Screenshot_20220420_160054.jpg',target_size=(512,512,3))
y_pred = model.predict(test_images)

# n = test_targets[0]
# image = cv2.imread('/content/drive/MyDrive/PennFudanPed/Test/FudanPed00011.png')
# cv2.rectangle(image, (n[0],n[1]), (n[2],n[3]), (255,0,0), 2)
# cv2.rectangle(image, (m[0],m[1]), (m[2],m[3]), (0,255,0), 2)
# cv2_imshow(image)
# cv2.waitKey(0)
