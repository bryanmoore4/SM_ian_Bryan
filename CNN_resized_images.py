#Visualize Fishery data
#Bryan Moore 1/1/2017
import os
import keras.utils
from collections import defaultdict
import PIL
import SimpleCV
from scipy.misc import imread
import numpy as np
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D



def resize_photo(fish_type,location,trainq):
    #   output = logo.edges(t1=350)
#    logo = SimpleCV.Image(location)
    output = PIL.Image.open(location)
    output = output.resize((800,600))
    output2 = np.array(output).astype('float32')
########## IF YOU WANT TO SAVE THE NEW RESIZED IMAGES INTO SEPERATE DIRECTORY UNCOMMENT BELOW #################
#    splitname = location.split("/")[3]
#    splitname2 = splitname.split(".")[0]
#    savename = str(fish_type+"_"+splitname2+"_800_600_scale.jpg")
#    os.chdir(scaled_directory)
#    if os.path.isdir("train") == False:
#        os.mkdir("train")
#    if os.path.isdir("test") == False:
#        os.mkdir("test")
#    if trainq == True:
#        os.chdir("train")
#        output.save(savename)
#        os.chdir("..")
#    if trainq == False:
#        os.chdir("test")
#        output.save(savename)
#        os.chdir("..")
###########################################################################################################
    os.chdir(where_to_start)
    return output2

def create_dataset(fishery_dict_list):
    #Converts picture into numpy array and it's label, also puts it into the training and testing allocations
    labels_to_int = {}
    label_int = 0
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    os.chdir(where_to_start)
    for i,line in enumerate(fishery_dict_list):
        for j,line2 in enumerate(fishery_dict_list[line]):
                #Training loop
                if j < data_size_label*training_portion:
                    x_train.append(resize_photo(line,line2,trainq=True))
                    if line not in labels_to_int:
                        labels_to_int[line] = label_int
                        label_int += 1
                    y_train.append(labels_to_int[line])
    #                x_data.append((line,new_photo_name))
                #Testing loop
                if j >= data_size_label*training_portion and j <= data_size_label:
                    x_test.append(resize_photo(line,line2,trainq=False))
                    if line not in labels_to_int:
                        labels_to_int[line] = label_int
                        label_int += 1
                    y_test.append(labels_to_int[line])
    #                y_data.append((line,new_photo_name))
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    return x_train,y_train,x_test,y_test

#Creating full dictionary of types of fish and their image location
data_augmentation = True
fishery_dict_list = defaultdict(list)
fish_type = []
folder = "train/train"
data_size_label = 10
training_portion = 0.8
scaled_directory = "scaled_data_800_600"
nb_classes = 6
batch_size = 4
nb_epoch = 5
if os.path.isdir(scaled_directory) == False:
    os.mkdir(scaled_directory)
where_to_start = os.getcwd()    
for filename in os.listdir(folder):
    if len(filename) != 3:
        continue
    if len(filename) < 3:
        continue
    if filename not in fishery_dict_list:
        count = 0
        folder2 = str(folder+"/"+filename)
        for image_files in os.listdir(folder2):
            destination = str(folder2+"/"+image_files)
            fishery_dict_list[filename].append(destination)
            count+=1

#gathering all the data and putting them into numpy arrays and their label, then pushing into
#training and testing datasets
x_train,y_train,x_test,y_test = create_dataset(fishery_dict_list)




###############################################################################################
#               BUILDING THE CNN MODEL
###############################################################################################


model = Sequential()

model.add(Convolution2D(5, 4, 3, border_mode='same',
                        input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Convolution2D(5, 4, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(5, 4, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(5, 4, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
if not data_augmentation:
    print("Not using Data Augmentation")
    model.fit(x_train, y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                        batch_size=batch_size),
                        samples_per_epoch=x_train.shape[0],
                        nb_epoch=nb_epoch,
                        validation_data=(x_test, y_test))