import os
from torch.utils.data import Dataset
from PIL import Image

class Flickr8kDataset(Dataset):

	def __init__(self, dataset_folder='./dataset'):
		self.images_folder = os.path.join(dataset_folder,'images')
		self.captions_file = open(os.path.join(dataset_folder,'captions.txt'),'r')
		self.dataset = [line.strip() for line in self.captions_file.readlines()[1:]]
		self.captions_file.close()

	def read_image(self,filename):
		image_path = os.path.join(self.images_folder, filename)
		image = Image.open(image_path)
		return image

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self,idx):
		filename, caption = self.dataset[idx].split(',')
		image = self.read_image(filename)
		label = caption
		return image,label


if __name__ == "__main__":
	dataset = Flickr8kDataset()
	dataset.__getitem__(15)