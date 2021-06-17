import pytest

import os
import torch

from torchvision import transforms

import numpy as np
import pandas as pd
import json
import sys
sys.path.insert(1, './')    ## Este path funciona en caso de ejecutar este fichero desde el directorio base del proyecto, desde el cual cuelgan todos los módulos del proyecto.

from dataset.main import Flickr8kDataset
from train import split_subsets

CONFIGURATION = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cwd = os.getcwd()

with open(os.path.join(cwd, 'config.json')) as f:
	CONFIGURATION = json.load(f)

hparams = CONFIGURATION['HPARAMS']
hparams['DEVICE'] = device

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

	
	## FIRST WE PERFORM THE SPLIT OF THE DATABASE USING ALL CAPTIONS AND PERFORM SOME TESTS
	
	train_split_all, test_split_all = split_subsets(dataset,all_captions=True)

	# Checking number of indices/samples for the case of all_captions=True (using all the captions per image)
	assert len(train_split_all)+len(test_split_all) == 40455, "The total number of samples in the datasets is different from 40.455!"
	all_indexes = train_split_all.indices + test_split_all.indices
	assert len(np.unique(all_indexes)) == 40455 , "The list with all the indexes of the database contain non unique indexes!!"

	#######     Checking that the image filenames repeat 5 times in training split that uses the 5 captions per image   #######


	# Create and fill img_files list with the image filenames corresponding to the indexes that have been used above for the TRAIN split. 
	img_files = []
	for i in train_split_all.indices:
		img_files.append([dataset.dataframe.iloc[i,0]])

	# GEt the unique values of img_files list. To do that I transfrom the list to a Numpy array and use the method unique() 
	pd_img_files = pd.DataFrame(img_files)
	unique_files_train =  pd.unique(pd_img_files[0])
	assert len(unique_files_train)*5==len(train_split_all) , "The total number of samples in the trainining split is not 5 times the number of unique image filenames!"

	# Let's check if every filename is present 5 times in the training split
	for filename in unique_files_train:
		x = img_files.count([filename])
		assert x==5 , f'The file {filename} is not present 5 times in the training split. It appears {x} times'


	#######     Checking that the image filenames repeat 5 times in test split that uses the 5 captions per image   #######

	# Create and fill img_files list with the image filenames corresponding to the indexes that have been used above for the TEST split. 
	img_files = []
	for i in test_split_all.indices:
		img_files.append([dataset.dataframe.iloc[i,0]])

	# GEt the unique values of img_files list. To do that I transfrom the list to a Numpy array and use the method unique() 
	pd_img_files = pd.DataFrame(img_files)
	unique_files_test =  pd.unique(pd_img_files[0])
	assert len(unique_files_test)*5==len(test_split_all) , "The total number of samples in the test split is not 5 times the number of unique image filenames!"

	# Let's check if every filename is present 5 times in the test split
	for filename in unique_files_test:
		x = img_files.count([filename])
		assert x==5 , f'The file {filename} is not present 5 times in the test split. It appears {x} times'

	# Check that the filenames in train does not appear in the test list.

	assert len(unique_files_train)+len(unique_files_test)==len(np.unique(np.concatenate((unique_files_train, unique_files_test)))) , 'One or more files in the training set is present in the test set!'

	## WE NOW PERFORM THE SPLIT OF THE DATABASE USING ONE CAPTION FOR IMAGE IN TRAINING AND 5 IN TEST. WE DO THE CHECKING

	train_split, test_split = split_subsets(dataset,all_captions=False)

	# Checking number of indices/samples for the case of all_captions=False (using only one caption per image)
	assert len(train_split)+len(test_split)/5 == 8091 , "The total number of samples in the datasets is different from 8091!"
	all_indexes = train_split.indices + test_split.indices
	assert len(np.unique(all_indexes)) ==  len(all_indexes), "The list with all the indexes of the database contain non unique indexes!!"

	#######   Now we check that the image filenames repeat 1 times in training split that uses 1 caption per image   #######

	# Create and fill img_files list with the image filenames corresponding to the indexes that have been used above for the TRAIN split. 
	img_files = []
	for i in train_split.indices:
		img_files.append([dataset.dataframe.iloc[i,0]])

	# GEt the unique values of img_files list. To do that I transfrom the list to a Numpy array and use the method unique() 
	pd_img_files = pd.DataFrame(img_files)
	unique_files_train =  pd.unique(pd_img_files[0])
	assert len(unique_files_train)==len(train_split) , "All filenames in the trainining split are uniqe()"

	# Let's check if every filename is present 1 times in the training split
	for filename in unique_files_train:
		x = img_files.count([filename])
		assert x==1 , f'The file {filename} is not present only one time in the training split. It appears {x} times'

	#######   Now we check that the image filenames repeat 5 times in test split that uses 1 caption per image   #######
	# Create and fill img_files list with the image filenames corresponding to the indexes that have been used above for the TEST split. 
	img_files = []
	for i in test_split.indices:
		img_files.append([dataset.dataframe.iloc[i,0]])

	# GEt the unique values of img_files list. To do that I transfrom the list to a Dataframe and use the method unique() 
	pd_img_files = pd.DataFrame(img_files)
	unique_files_test =  pd.unique(pd_img_files[0])	
	assert len(unique_files_test)*5==len(test_split) , "The total number of samples in the test split is not 5 times the number of unique image filenames!"

	# Let's check if every filename is present 5 times in the test split
	for filename in unique_files_test:
		x = img_files.count([filename])
		assert x==5 , f'The file {filename} is not present 5 times in the test split. It appears {x} times'

	# Check that the filenames in train does not appear in the test list.

	assert len(unique_files_train)+len(unique_files_test)==len(np.unique(np.concatenate((unique_files_train, unique_files_test)))) , 'One or more files in the training set is present in the test set!'

if __name__ == "__main__":
	test_main()
