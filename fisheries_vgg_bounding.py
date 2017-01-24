#%matplotlib inline
from __future__ import division, print_function
import utils; reload(utils)
from utils import *
from shutil import copyfile,move,copy
from vgg16bn import Vgg16BN


#Make sure to put this file in the directory that holds the test and training data
path_to_datasets = str(os.getcwd() + "/train/")
path_to_data = str(os.getcwd() + "/train/train")
path_to_train = str(os.getcwd() + "/train/train")
path_to_valid = str(os.getcwd() + "/train/valid")
os.chdir(path_to_data)
train_directory = "../train/"
valid_directory = "../valid/"
s_train_directory = "sample/train"
s_valid_directory = "sample/valid"
batch_size=64

if os.path.isdir("../../test_stg1/test_stg1") == True:
    os.mkdir("../test")
    move("../../test_stg1/test_stg1","../test/")
    os.rename("../test/test_stg1","../test/test")

    
if os.path.isdir(valid_directory) == False:
    os.mkdir(valid_directory)
    g = glob('*')
    for d in g: os.mkdir(valid_directory+d)
    
    g = glob('*/*.jpg')
    shuf = np.random.permutation(g)
    for i in range(500): os.rename(shuf[i], valid_directory + shuf[i])
    
    os.mkdir("../sample")
    os.mkdir("../sample/train")
    os.mkdir("../sample/valid")
    
    g = glob('*')
    for d in g: 
        os.mkdir('../sample/train/'+d)
        os.mkdir('../sample/valid/'+d)
    
    
    g = glob('*/*.jpg')
    shuf = np.random.permutation(g)
    for i in range(400): copyfile(shuf[i], '../sample/train/' + shuf[i])
    
    os.chdir("../valid")
    
    g = glob('*/*.jpg')
    shuf = np.random.permutation(g)
    for i in range(200): copyfile(shuf[i], '../sample/valid/' + shuf[i])
    
    
    os.mkdir("../results")
    os.mkdir("../sample/results")
    os.chdir("../..")
        
    print(os.getcwd())


batches = get_batches(path_to_train, batch_size=batch_size)
val_batches = get_batches(path_to_valid, batch_size=batch_size*2, shuffle=False)
(val_classes, trn_classes, val_labels, trn_labels, 
    val_filenames, filenames, test_filenames) = get_classes(path_to_datasets)

raw_filenames = [f.split('/')[-1] for f in filenames]
raw_test_filenames = [f.split('/')[-1] for f in test_filenames]
raw_val_filenames = [f.split('/')[-1] for f in val_filenames]

if os.path.isdir("../results/trn.dat")==False:
    test = get_data(path_to_datasets+'test')
    trn = get_data(path_to_datasets+'train')
    val = get_data(path_to_datasets+'valid')
    save_array(path_to_datasets+'results/trn.dat', trn)
    save_array(path_to_datasets+'results/val.dat', val)
    save_array(path_to_datasets+'results/test.dat', test)
else:
    trn = load_array(path_to_datasets+'results/trn.dat')
    val = load_array(path_to_datasets+'results/val.dat')
    test = load_array(path_to_datasets+'results/test.dat')
#
#
if os.path.isfile("../model/ft.h5") == True:
    model = vgg_ft_bn(8)
    model.load_weights("../model/ft1.h5")
    print("loaded pre-trained weights from file")
else:
    model = vgg_ft_bn(8)
    model.compile(optimizer=Adam(1e-3),
           loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(trn, trn_labels, batch_size=batch_size, nb_epoch=3, validation_data=(val, val_labels))
    model.save_weights('../model/ft1.h5')
#
#
## Rebuild the last layer of the model, keep the rest of the model
#conv_layers,fc_layers = split_at(model, Convolution2D)
#conv_model = Sequential(conv_layers)
#conv_feat = conv_model.predict(trn)
#conv_val_feat = conv_model.predict(val)
#conv_test_feat = conv_model.predict(test)
#
#if os.path.isdir("../results/conv_val_feat.dat") == False:
#    save_array(path_to_datasets+'results/conv_val_feat.dat', conv_val_feat)
#    save_array(path_to_datasets+'results/conv_feat.dat', conv_feat)
#    save_array(path_to_datasets+'results/conv_test_feat.dat', conv_test_feat)
#else:
#    conv_feat = load_array(path_to_datasets+'results/conv_feat.dat')
#    conv_val_feat = load_array(path_to_datasets+'results/conv_val_feat.dat')
#    conv_test_feat = load_array(path_to_datasets+'results/conv_test_feat.dat')
