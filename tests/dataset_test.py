import os
import torch
from torch.utils.data.dataset import Subset
from torchvision import transforms
import random
import numpy as np
import pandas as pd
import sys
sys.path.insert(1, './')    ## Este path funciona en caso de ejecutar este fichero desde el directorio base del proyecto, desde el cual cuelgan todos los módulos del proyecto.

from dataset.main import Flickr8kDataset
from model.main import ImageCaptioningModel
import json

CONFIGURATION = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cwd = os.getcwd()

with open(os.path.join(cwd, 'config.json')) as f:
    CONFIGURATION = json.load(f)

hparams = CONFIGURATION['HPARAMS']
hparams['DEVICE'] = device


def split_subsets(dataset,train_percentage=0.8,all_captions=True):
	"""
	Performs the split of the dataset into Train and Test
	"""	
	assert len(dataset) == 40455, "The dataset object doesn't contain 40.455 entries!"

	if all_captions==True:

		# Get a list of the first indexes for each image in the dataset and convert to a numpy array  
		all_indexes = np.array([*range(0,len(dataset))])
		# Reshape the array so we can shuffle indexes in chunks of 5
		all_indexes_mat = all_indexes.reshape(-1,5)
		np.random.shuffle(all_indexes_mat)
		all_indexes_shuffled = all_indexes_mat.flatten()

		assert len(all_indexes_shuffled) == 40455, "The list with all the indexes that will be splitted into train and test doesn't contain 40455 indexes!"
		assert len(np.unique(all_indexes_shuffled)) == 40455 , "The list with all the indexes of the database contain non unique indexes!!"

		# Get the number of images for train and the rest are for test
		num_train_imgs = int(len(all_indexes_shuffled)/5*train_percentage)

		# Create the subsets for train and test
		train_split =  Subset(dataset,all_indexes_shuffled[0:num_train_imgs*5].tolist())
		test_split =  Subset(dataset,all_indexes_shuffled[num_train_imgs*5:].tolist())	

		#######     Checking that the image filenames repeat 5 times in the list with the indexes used to creat the TRAIN subset   #######

		# Create and fill img_files list with the image filenames corresponding to the indexes that have been used above for the TRAIN split. 
		img_files = []
		for i in all_indexes_shuffled[0:num_train_imgs*5]:
			img_files.append([dataset.dataframe.iloc[i,0]])

		# GEt the unique values of img_files list. To do that I transfrom the list to a Dataframe and use the method unique() 
		df_aux = pd.DataFrame(img_files)
		unique_files =  pd.unique(df_aux[0])
		assert len(unique_files)*5==len(all_indexes_shuffled[0:num_train_imgs*5]) , "Files don't repeat 5 times each!"

		for filename in unique_files:
			x = img_files.count([filename])
			assert x==5 , f'The file {filename} is not present 5 times in the training split. Only {x} times'		

		# Now for the TEST split. Checking that the image filenames repeat 5 times in the list with the indexes used to creat the TEST subset   ##

		# Create and fill img_files list with the image filenames corresponding to the indexes that have been used above for the TEST split. 
		img_files = []
		for i in all_indexes_shuffled[num_train_imgs*5:]:
			img_files.append([dataset.dataframe.iloc[i,0]])

		# GEt the unique values of img_files list. To do that I transfrom the list to a Dataframe and use the method unique() 
		df_aux = pd.DataFrame(img_files)
		unique_files =  pd.unique(df_aux[0])
		assert len(unique_files)*5==len(all_indexes_shuffled[num_train_imgs*5:]) , "Files don't repeat 5 times each!"

		for filename in unique_files:
			x = img_files.count([filename])
			assert x==5 , f'The file {filename} is not present 5 times in the test split. Only {x} times'		

	else:
		# Create a list with the first indexes and shuffle it
		all_first_index = [*range(0,len(dataset),5)]
		random.shuffle(all_first_index)

		assert len(all_first_index) == len(dataset)/5, "Variable all_first_index does not containt the expected 8091 indixes"

        # Calculate the number of training images from the train_percentage parameter
		num_train_imgs = int(len(all_first_index)*train_percentage)

		train_split =  Subset(dataset,all_first_index[0:num_train_imgs])
		test_split =  Subset(dataset,all_first_index[num_train_imgs:])	
		
		#######   Checking if there are repeated images filenames in the TRAIN list
		 
		# Create and fill img_files list with the image filenames corresponding to the indexes that have been used above for the TRAIN split. 

		img_files = []
		for i in all_first_index[0:num_train_imgs]:
			img_files.append([dataset.dataframe.iloc[i,0]])

		# GEt the unique values of img_files list. To do that I transfrom the list to a Dataframe and use the method unique()
		df_aux = pd.DataFrame(img_files)
		unique_values_train =  pd.unique(df_aux[0])

		assert len(unique_values_train)==len(all_first_index[0:num_train_imgs]) , "There are repeated image filenames in train split!"

		# Now for the TEST split. Checking if there are any repeated images filenames in the TEST list
		
		# Create and fill img_files list with the image filenames corresponding to the indexes that have been used above for the TEST split. 
		img_files = []
		for i in all_first_index[num_train_imgs:]:
			img_files.append([dataset.dataframe.iloc[i,0]])
		
		# GEt the unique values of img_files list. To do that I transfrom the list to a Dataframe and use the method unique()
		df_aux = pd.DataFrame(img_files)
		unique_values_test =  pd.unique(df_aux[0])

		assert len(unique_values_test)==len(all_first_index[num_train_imgs:]) , "There are repeated image filenames in test split!"

	return train_split,test_split

def test_main():

	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Resize((hparams['IMAGE_SIZE'],hparams['IMAGE_SIZE'])),
		# The normalize parameters depends on the model we're gonna use
		# If we apply transfer learning from a model that used ImageNet, then
		# we should use the ImageNet values to normalize the dataset.
		# Otherwise we could just normalize the values between -1 and 1 using the 
		# standard mean and standard deviation
		transforms.Normalize(mean=hparams['IMAGE_NET_MEANS'],std=hparams['IMAGE_NET_STDS']),
	])
	dataset = Flickr8kDataset(dataset_folder=CONFIGURATION['DATASET_FOLDER'], transform=transform,
								reduce=hparams['REDUCE_VOCAB'], vocab_max_size=hparams['VOCAB_SIZE'])

	## Perform the split of the dataset
	
	train_split, test_split = split_subsets(dataset,all_captions=True)
	train_split, test_split = split_subsets(dataset,all_captions=False)

if __name__ == "__main__":
	test_main()
