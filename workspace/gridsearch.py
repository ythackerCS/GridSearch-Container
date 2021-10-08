import os 
import pandas as pd #for data analysis 
import matplotlib.pyplot as plt 
import cv2 
import numpy as np 
import math 
import pydicom as pydicom
import tensorflow as tf 
import tensorflow_addons as tfa
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import gdcm
import random
import scipy.ndimage
import collections
import imblearn
import numpy


from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras import utils as np_utils
from keras.utils.np_utils import to_categorical
from random import seed
from random import random
from random import randint
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
from tensorflow.keras import models


## ---------- Data set up functions, balancing data, and spliting dataset ------------ ######

#this function extracts the pixel_data and resizes it to a user defined size defult is 50, and the associated vector for classification is generated 
#NOTE: Slight changes are made to all of the code to make futher improvements using better libraries 

def getImagesAndLabels(imageArray, labelArray, img_px_size=50, visualize=False):
    np.random.seed = 1
    images = []
    labels = []
    uids = []
    idx = 0
    print("getting images and labels")

    for file, mortality in tqdm(zip(imageArray.iteritems(), labelArray.iteritems()),total = len(imageArray)): 
        uid = file[1]
        label=mortality[1]
        path = uid
        image = pydicom.read_file(path)
        if image.Modality != "SR": 
            if "PixelData" in image: 
                idx += 1
                resized_image = cv2.resize(np.array(image.pixel_array),(img_px_size,img_px_size))
                value = randint(1, 10)
                factor = random()
                if value == 3:
                    fig, ax = plt.subplots(1,2)
                    ax[0].imshow(resized_image)
                    resized_image = np.fliplr(resized_image)
                    ax[1].imshow(resized_image)

                    #NOTE: within a docker container you will not be able to see these visualization 

                    #uncomment this line if you would like to see what the image looks like when flipped accross the y axis 
                    # plt.show()


                #this set of code is commented out as visuilization is not possible within a docker container but if you run this seperatly or within jupyter you are able to visulize every 15th image, change 15 to what ever number you like if you want to visulize more or less 

                if visualize: 
                    #every 15th image is "visualized" changing the 15 will allow you view every xth image 
                    if idx%15==0:
                        fig = plt.figure() 
                        plt.imshow(resized_image)
                        # plt.show()
                        
                images.append(resized_image)
                labels.append(label)
                uids.append(uid)
    print("total subjects avilable: ", idx)
    
    print("lenth of images", len(images))
    
    return images, labels, uids


#this function will balance data however compared to the TrainModel-Container, after gaining futher understanding, test data is not blanced to mimic real world work. Credit for help understanding this Sotiras, A. Assistant Professor of Radiology @WASHU  

#as the dataset was imbalanced, balancing tehniques were applied, in this case the number of dicom for each class is counted and then balanced according to user's preference, it can either be undersampeled or over sampeled 

def balanceData(imageArray, labelArray, underSample = False,):
#     print(imageArray, labelArray)
    concatinatedArrray = pd.concat([imageArray, labelArray], axis=1)
    
    count_class_0, count_class_1 = concatinatedArrray.mortality.value_counts()
    df_class_0 = concatinatedArrray[concatinatedArrray['mortality'] == 0]
    df_class_1 = concatinatedArrray[concatinatedArrray['mortality'] == 1]
    
    print("alive", len(df_class_0), "dead", len(df_class_1))
    
#     print("before balancing")
    concatinatedArrray.mortality.value_counts().plot(kind='bar', title='before balancing');
    
    #undersampleling of data is done if user cooses to under sample 
    if underSample: 
        df_class_0_under = df_class_0.sample(count_class_1)
        df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)

        print('Random under-sampling:')
#         print(df_test_under.mortality.value_counts())

#         print("after balancing")
        df_test_under.mortality.value_counts().plot(kind='bar', title='after balancing_undersample');
        total_data = pd.concat([df_class_0_under, df_class_1])
        
#         print(len(total_data))
    
    #over sampleing is done if user does not check undersample 
    else: 
        df_class_1_over = df_class_1.sample(count_class_0, replace=True)
        df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)

        print('Random over-sampling:')
#         print(df_test_over.mortality.value_counts())

#         print("after balancing")
        df_test_over.mortality.value_counts().plot(kind='bar', title='after balancing_oversample');
        total_data = pd.concat([df_class_0, df_class_1_over])
        
#         print(len(total_data))
        
    return total_data.path, total_data.mortality, total_data
        
#this function will split the data in to train,validation, and test datasets steps are as follows: 
    #1 user provides testSize, which will split the orgninal data set in to 1-x% training and x% "test dataset"
    #2 the "test dataset" is then split again in half for validation and half an actual test dataset

def splitData(px_size, visulize = False, testSize = 0.30, randState = 50, underSamp=False, numClasses=2):
    count_class_0, count_class_1 = df_train.mortality.value_counts()
    
    
    #getting classes counts 
    df_class_0 = df_train[df_train['mortality'] == 0]
    df_class_1 = df_train[df_train['mortality'] == 1]
    
    
    total_data = pd.concat([df_class_0, df_class_1])
    
    #seperating data into images and labels data set 
    images = total_data.path
    labels = total_data.mortality
    
    
    #spliting images and labeles into train and test data set 
    image_train, image_test, label_train, label_test = train_test_split(images, labels, test_size=testSize, random_state=randState)
    
    #test data set split into half for validation half for testing
    image_val, image_test, label_val, label_test = train_test_split(image_test, label_test, test_size=0.5, random_state=randState)
    
    
    #balancing training and testing data sets 
    image_train, label_train, total_data_train = balanceData(image_train, label_train, underSample = underSamp)
   
    image_test, label_test, total_data_test = balanceData(image_test, label_test, underSample = underSamp)
    

    #extracting pixel array for training, testing and validation ds
    image_train, label_train, uids_train = getImagesAndLabels(image_train, label_train, img_px_size=px_size, visualize=visulize)
    image_test, label_test, uids_test = getImagesAndLabels(image_test, label_test, img_px_size=px_size, visualize=visulize)
    image_val, label_val, uids_val = getImagesAndLabels(image_val, label_val, img_px_size=px_size, visualize=visulize)
    
    
    #printing counts of distributions of each class
    print("Distribution of classes in train ds")
    unique, counts = np.unique(label_train, return_counts=True)
    collections.Counter(label_train)
    print(dict(zip(unique, counts)))
    
    print("Distribution of classes in test ds")
    unique, counts = np.unique(label_test, return_counts=True)
    collections.Counter(label_test)
    print(dict(zip(unique, counts)))
    
    print("Distribution of classes in val ds")
    unique, counts = np.unique(label_val, return_counts=True)
    collections.Counter(label_val)
    print(dict(zip(unique, counts)))
    label_test = tf.keras.utils.to_categorical(label_test, numClasses)
    
    print(image_test[1].shape)
    print(label_test[1].shape)
    
    
    #this is an improvement from TrainModel-Container the to_categorical method was used to make it adaptible to more than 2 classes
    label_train = tf.keras.utils.to_categorical(label_train, numClasses)

    label_val = tf.keras.utils.to_categorical(label_val, numClasses)
    

    return image_train, image_test, image_val, label_train, label_test, label_val, uids_train, uids_test, uids_val

## ------------------ END data ulities -----------------------##


df_train = pd.read_csv('/csvLocation/{0}.csv'.format("mortality"))
image_train, image_test, image_val, label_train, label_test, label_val, uids_train, uids_test, uids_val = splitData(px_size=50, visulize=False, testSize = .2, randState = 50, underSamp = False, numClasses=2)



##---Statsitics functions---#
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
#--------------------------------#

## ---- MODEL architecture (CNN) ----## 
#this was obtained and developed with the help of a gradstudent Satrajit Chakrabarty @WASHU, the paper used is: Isensee F, Kickingereder P, Wick W, Bendszus M, Maier-Hein KH. Brain tumor segmentation and radiomics survival prediction: Contribution to the brats 2017 challenge. In: International MICCAI Brainlesion Workshop. Springer; 2017. p. 287–97.
# Function to create model, required for KerasClassifier
def create_context_module(input_layer, n_level_filters, dropout_rate=0.3, regularizer = None):
    conv1 = create_convolution_block(input_layer=input_layer, n_filters=n_level_filters, regularizer = regularizer)
    dropout = tf.keras.layers.SpatialDropout2D(rate=dropout_rate)(conv1)
    conv2 = create_convolution_block(input_layer=dropout, n_filters=n_level_filters, regularizer = regularizer)
    return conv2

def create_convolution_block(input_layer, n_filters, name=None, kernel=3, padding='SAME', strides=1, regularizer=None):
    layer = tf.keras.layers.Conv2D(n_filters, kernel, padding=padding, strides=strides, name=name, kernel_regularizer=regularizer)(input_layer)
    layer = tfa.layers.InstanceNormalization(axis=1)(layer) 
    return tf.keras.layers.LeakyReLU()(layer)

def isensee2017_classification_2d(input_shape=[50,50,1],
                                nb_classes = 2,
                                n_base_filters=16,
                                context_dropout_rate=0.3,
                                gap_dropout_rate=0.4,                                                   
                                regularizer=None,
                                learnrate = 0.00001):
#     K.clear_session()
    inputs = tf.keras.Input(input_shape)
    depth = 5
    filters = [(2 ** i) * n_base_filters for i in range(depth)]
    # level 1: input --> conv_1 (stride = 1) --> context_1 --> summation_1
    conv_1 = create_convolution_block(inputs, filters[0], regularizer=regularizer)
    context_1 = create_context_module(conv_1, filters[0], dropout_rate=context_dropout_rate, regularizer=regularizer)
    summation_1 = tf.keras.layers.Add()([conv_1, context_1])
    # level 2: summation_1 --> conv_2 (stride = 2) --> context_2 --> summation_2
    conv_2 = create_convolution_block(summation_1, filters[1], strides=2, regularizer=regularizer)
    context_2 = create_context_module(conv_2, filters[1], dropout_rate=context_dropout_rate, regularizer=regularizer)
    summation_2 = tf.keras.layers.Add()([conv_2, context_2])
    # level 3: summation_2 --> conv_3 (stride = 2) --> context_3 --> summation_3
    conv_3 = create_convolution_block(summation_2, filters[2], strides=2, regularizer=regularizer)
    context_3 = create_context_module(conv_3, filters[2], dropout_rate=context_dropout_rate, regularizer=regularizer)
    summation_3 = tf.keras.layers.Add()([conv_3, context_3])
    # level 4: summation_3 --> conv_4 (stride = 2) --> context_4 --> summation_4
    conv_4 = create_convolution_block(summation_3, filters[3], strides=2, regularizer=regularizer)
    context_4 = create_context_module(conv_4, filters[3], dropout_rate=context_dropout_rate, regularizer=regularizer)
    summation_4 = tf.keras.layers.Add()([conv_4, context_4])
    # level 5: summation_4 --> conv_5 (stride = 2) --> context_5 --> summation_5
    conv_5 = create_convolution_block(summation_4, filters[4], strides=2, regularizer=regularizer)
    context_5 = create_context_module(conv_5, filters[4], dropout_rate=context_dropout_rate, regularizer=regularizer)
    summation_5 = tf.keras.layers.Add()([conv_5, context_5])
    clsfctn_op_GAP_summation_5 = tf.keras.layers.GlobalAveragePooling2D()(summation_5)
    if gap_dropout_rate:
        aggregated_maps = tf.keras.layers.Dropout(rate=gap_dropout_rate)(clsfctn_op_GAP_summation_5)
    clsfctn_Dense = tf.keras.layers.Dense(nb_classes, name="Dense_without_softmax", kernel_regularizer=regularizer)(aggregated_maps)
    clsfctn_op = tf.keras.layers.Activation('softmax', name="clsfctn_op")(clsfctn_Dense)
    model_clsfctn = tf.keras.Model(inputs=inputs, outputs=clsfctn_op)
    model_clsfctn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learnrate), loss='binary_crossentropy', metrics=['acc',f1_m,precision_m])
    return model_clsfctn

## -------- END Model architecture --------- ##


##------ GRID Search set up and run ------##

# fix random seed for reproducibility
np.random.seed = 1
# load dataset

# split into input (X) and output (Y) variables
X = numpy.array(image_train)
Y = numpy.array(label_train)


# create model
model = KerasClassifier(build_fn=isensee2017_classification_2d, verbose=0)


# define the grid search parameters
batch_size = [5, 10, 15]
epochs = [10,20,30,50,100]
learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
regularizers = [tf.keras.regularizers.l1(0.01), tf.keras.regularizers.l2(0.01)]


#create grid search param dict 
param_grid = dict(batch_size=batch_size, epochs=epochs, regularizer=regularizers, learnrate=learn_rate, context_dropout_rate=dropout_rate,gap_dropout_rate=dropout_rate)

#initialize gridsearch 
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3, verbose=10)
grid_result = grid.fit(X, Y)


## ----------- End gridsearch params and run ---------- ##


## ----- Printing results and saving them ----###
# summarize results and print them into a params.txt within results folder for easy viewing 

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

with open("/results/params.txt", "w") as f:
	for mean, stdev, param in zip(means, stds, params):
    		gridsearchline = "%f (%f), with: %r" % (mean, stdev, param)
    		f.write(gridsearchline + "\n")
    		print(gridsearchline)

	f.write("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))    

## --------- End results componenets ----------- ## 