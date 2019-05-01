import os
from shutil import copyfile


dtypes = ["train", "trainval","val","test"]


classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
		"cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
		"sheep", "sofa", "train", "tvmonitor"]

txt_files_root = "/media/jedrzej/Seagate/DATA/VOC2012/ImageSets/Main/"
images_root = "/media/jedrzej/Seagate/DATA/VOC2012/JPEGImages/"
new_images_root = "/media/jedrzej/Seagate/DATA/VOC2012/PyTorch/"

for dtype in dtypes:
	print
	print dtype
	text_file = open(txt_files_root+dtype+".txt", "r")
	all_images = text_file.readlines()
	for c in classes:
		print c
		directory = new_images_root+dtype+"/"+c
		if not os.path.exists(directory):
    			os.makedirs(directory)
		text_file = open(txt_files_root+c+"_"+dtype+".txt", "r")
		current_images = text_file.readlines()
		for i in xrange(len(current_images)):
			info = current_images[i].split()
			if(info[-1]==str(1)): 
				# copy the image to the dst folder
				copyfile(images_root+info[0]+'.jpg', directory+'/'+info[0]+'.jpg')
		
