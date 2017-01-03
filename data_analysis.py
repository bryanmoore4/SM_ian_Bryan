#Visualize Fishery data
#Bryan Moore 1/1/2017
import os
from collections import defaultdict
import PIL
import SimpleCV

fishery_dict_list = defaultdict(list)
fish_type = []
folder = "train/train"

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
            if count == 10:
                break

for i,line in enumerate(fishery_dict_list):
    for j,line2 in enumerate(fishery_dict_list[line]):
        if i ==2 and j ==1:
            print line
            logo = SimpleCV.Image(line2)
            output = logo.edges(t1=350)
            splitname = line2.split("/")[3]
            splitname2 = splitname.split(".")[0]
            savename = str(line+"_"+splitname2+"_edges.jpg")
            output.save(savename)
            output.live()
 