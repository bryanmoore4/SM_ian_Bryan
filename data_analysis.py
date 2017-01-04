#Visualize Fishery data
#Bryan Moore 1/1/2017
import os
from collections import defaultdict
import PIL
import SimpleCV

fishery_dict_list = defaultdict(list)
fish_type = []
folder = "train/train"
data_size_label = 10
training_portion = 0.8
scaled_directory = "scaled_data_800_600"
if os.path.isdir(scaled_directory) == False:
    os.mkdir(scaled_directory)
where_to_start = os.getcwd()

def resize_photo(fish_type,location,trainq):
    logo = SimpleCV.Image(location)
#   output = logo.edges(t1=350)
    output = logo.resize(800,600)
    splitname = location.split("/")[3]
    splitname2 = splitname.split(".")[0]
    savename = str(fish_type+"_"+splitname2+"_800_600_scale.jpg")
    os.chdir(scaled_directory)
    if os.path.isdir("train") == False:
        os.mkdir("train")
    if os.path.isdir("test") == False:
        os.mkdir("test")
    if trainq == True:
        os.chdir("train")
        output.save(savename)
        os.chdir("..")
    if trainq == False:
        os.chdir("test")
        output.save(savename)
        os.chdir("..")
    os.chdir(where_to_start)
    return savename
#    output.live()



#Creating full dictionary of types of fish and there image location
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
#            if count == 10:
#                break

#Now putting them in tuples (type of fish, file location), for a given size and in training and test lists
x_data = []
y_data = []
os.chdir(where_to_start)
for i,line in enumerate(fishery_dict_list):
    for j,line2 in enumerate(fishery_dict_list[line]):
            #Training loop
            if j < data_size_label*training_portion:
                new_photo_name = resize_photo(line,line2,trainq=True)
                x_data.append((line,new_photo_name))
            #Testing loop
            if j >= data_size_label*training_portion and j <= data_size_label:
                new_photo_name = resize_photo(line,line2,trainq=False)
                y_data.append((line,new_photo_name))
