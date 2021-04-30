import os
from torch.utils.data import Dataset

class Flickr8kDataset(Dataset):
	def __init__(self, dataset_folder='./dataset'):
		self.__folder = dataset_folder

	def __len__(self):
		pass

	def __getitem__(self,idx):
		image = None
		label = None

		return image,label