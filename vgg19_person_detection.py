!pip install tensorflow_addons
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
import tensorflow_addons as tfa
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config = config)
print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))
# Paths of train, test and validation image data
train_path = '/content/drive/MyDrive/PennFudanPed/Train'
test_path = '/content/drive/MyDrive/PennFudanPed/Test'
validation_path = '/content/drive/MyDrive/PennFudanPed/Validation'

# creating a empty list and iterating over every files in image folder and converting them into array and appending them in the created list.
train_images = []
for filename in sorted(os.listdir(train_path)):
    train_image_fullpath = os.path.join(train_path, filename)
    train_img = load_img(train_image_fullpath, target_size=(224,224)) #default input size of the vgg models = (224,224)
    train_img_arr = img_to_array(train_img)
    train_images.append(train_img_arr)

# Converting list into numpy arrays
train_images = np.array(train_images)

# creating a empty list and iterating over every files in image folder and converting them into array and appending them in the created list.
val_images = []
for filename in sorted(os.listdir(validation_path)):
    val_image_fullpath = os.path.join(validation_path, filename)
    val_img = load_img(val_image_fullpath, target_size=(224,224)) #default input size of the vgg models = (224,224)
    val_img_arr = img_to_array(val_img)
    val_images.append(val_img_arr)
  
# Converting list into numpy arrays
val_images = np.array(val_images)

# creating a empty list and iterating over every files in image folder and converting them into array and appending them in the created list.

test_images = []
for filename in sorted(os.listdir(test_path)):
    test_image_fullpath = os.path.join(test_path, filename)
    test_img = load_img(test_image_fullpath, target_size=(224,224)) #default input size of the vgg models = (224,224)
    test_img_arr = img_to_array(test_img)
    test_images.append(test_img_arr)

# Converting list into numpy arrays
test_images = np.array(test_images)

# Paths of train, test and validation bounding box coordinates
train_path1 = '/content/drive/MyDrive/PennFudanPed/anno/Train'
test_path1 = '/content/drive/MyDrive/PennFudanPed/anno/Test'
validation_path1 = '/content/drive/MyDrive/PennFudanPed/anno/Validation'
import xml.etree.ElementTree as ET
# Creating a empty list and iterating over bb coordinates and appending them in the list.
train_targets = []
for filename in sorted(os.listdir(train_path1)):
  train_image_fullpath = os.path.join(train_path1, filename)
  tree=ET.parse(train_image_fullpath)
  root=tree.getroot()
  for neighbor in root.iter('bndbox'):
    xmin = int(neighbor.find('xmin').text)
    ymin = int(neighbor.find('ymin').text)
    xmax = int(neighbor.find('xmax').text)
    ymax = int(neighbor.find('ymax').text)
  train_targets.append((xmin,ymin,xmax,ymax))

# Converting list into numpy array
train_targets = np.array(train_targets)

# Creating a empty list and iterating over bb coordinates and appending them in the list.
val_targets = []
for filename in sorted(os.listdir(validation_path1)):
  val_image_fullpath = os.path.join(validation_path1, filename)
  tree=ET.parse(val_image_fullpath)
  root=tree.getroot()
  for neighbor in root.iter('bndbox'):
    xmin = int(neighbor.find('xmin').text)
    ymin = int(neighbor.find('ymin').text)
    xmax = int(neighbor.find('xmax').text)
    ymax = int(neighbor.find('ymax').text)
  val_targets.append((xmin,ymin,xmax,ymax))
# Converting list into numpy array
val_targets = np.array(val_targets)
# Creating a empty list and iterating over bb coordinates and appending them in the list.
test_targets = []
for filename in sorted(os.listdir(test_path1)):
  test_image_fullpath = os.path.join(test_path1, filename)
  tree=ET.parse(test_image_fullpath)
  root=tree.getroot()
  for neighbor in root.iter('bndbox'):
    xmin = int(neighbor.find('xmin').text)
    ymin = int(neighbor.find('ymin').text)
    xmax = int(neighbor.find('xmax').text)
    ymax = int(neighbor.find('ymax').text)
  test_targets.append((xmin,ymin,xmax,ymax))

# Converting list into numpy array
test_targets = np.array(test_targets)

####### VGG19
#default input size of the vgg models = (224,224)
image_size = [224,224]

#Loading the pretrained vgg19 model(pretrained on imagenet dataset) from keras
vgg19 = VGG19(input_shape=image_size+[3], weights = 'imagenet', include_top = False)

#Freezing the top layers in the model
for layer in vgg19.layers:
   layer.trainable = False

# Building the custom layers for bb regression on top of vgg19 model
out = vgg19.output
flat = Flatten()(out)
FCL1 = Dense(128,activation = 'relu')(flat)
FCL2 = Dense(32,activation = 'relu')(FCL1)
Out = Dense(4)(FCL2)
model = Model(inputs = vgg19.input, outputs = Out)

# Using Mean Squared Error loss and General Intersection over Union metric for the model.
model.compile(
    loss = tensorflow.keras.losses.MSE,
    optimizer = 'adam',
    metrics = [tfa.losses.GIoULoss()]
)
model.summary()

# Train the model for 20 epochs
trained_model = model.fit(train_images, train_targets,
             validation_data=(val_images, val_targets),
             batch_size=len(train_images),
             epochs=100,
             shuffle=True,
             verbose=1)

#### Save model
from tensorflow.keras.models import load_model
# model.save('/content/drive/MyDrive/PennFudanPed/model_vg19.h5')
model = load_model('/content/drive/MyDrive/PennFudanPed/model_vg19.h5')

import cv2
from google.colab.patches import cv2_imshow

# test_img=load_img('/content/drive/MyDrive/PennFudanPed/Test/IMG20220324205659.jpg',target_size=(224,224))
y_pred = model.predict(test_images)
m = y_pred[1]
n = test_targets[1]
image = cv2.imread('/content/drive/MyDrive/PennFudanPed/Test/FudanPed00012.png')
cv2.rectangle(image, (n[0],n[1]), (n[2],n[3]), (255,0,0), 2)
cv2.rectangle(image, (m[0],m[1]), (m[2],m[3]), (0,255,0), 2)
cv2_imshow(image)
cv2.waitKey(0)
# predicted green
# actual blue

test_images[0]
image = cv2.imread('/content/drive/MyDrive/PennFudanPed/Test/IMG20220324205659.jpg')
# cv2.rectangle(image, (n[0],n[1]), (n[2],n[3]), (255,0,0), 2)
cv2.rectangle(image, (m[0]*13.71,m[1]*18.28), (m[2]*13.71,m[3]*18.28, (0,255,0), 2)
cv2_imshow(image)
cv2.waitKey(0)
              
image = cv2.imread('/content/drive/MyDrive/PennFudanPed/Test/FudanPed00011.png')
n = test_targets[0]
cv2.rectangle(image, (n[0],n[1]), (n[2],n[3]), (255,0,0), 2)
cv2_imshow(image)
cv2.waitKey(0)
