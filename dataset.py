import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class Vocabulary_simple(self):
    def __init__(self):
        self.word_to_index = word_to_index

        self.index_to_word = {v:k for k, v in self.word_to_index.items()
    self.


class Flickr8kDataset(Dataset):

	def __init__(self, dataset_folder='./dataset',transform=None):
		super().__init__()
		self.transform = transform
		self.images_folder = os.path.join(dataset_folder,'images')
		self.dataframe = pd.read_csv(open(os.path.join(dataset_folder,'captions.txt'),'r'))

	def read_image(self,filename):
		"""
		Reads the image file based on the filename

		Args:
			filename (string): Filename that contains the image

		Returns:
			[Image]: Returns a PIL.Image object 
		"""
		image_path = os.path.join(self.images_folder, filename)
		image = Image.open(image_path)
		return image

	def __len__(self):
		return len(self.dataframe)

	def __getitem__(self,idx):
		"""
		Process the dataset according to our needs.
		The image needs to be resize and normalize.

		Args:
			idx (int): Index of the dataset element to be returned

		Returns:
			torch.Tensor: 	Processed image
			string:			Caption of the image
		"""
		filename, caption = self.dataframe.iloc[idx,:]
		image = self.read_image(filename)

		if self.transform:
			image = self.transform(image)

		return image, caption


if __name__ == "__main__":
	image_size = (128,128)
	batch_size = 128
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Resize(image_size),
		# The normalize parameters depends on the model we're gonna use
		# If we apply transfer learning from a model that used ImageNet, then
		# we should use the ImageNet values to normalize the dataset.
		# Otherwise we could just normalize the values between -1 and 1 using the 
		# standard mean and standard deviation
		# transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
		transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
	])
	dataset = Flickr8kDataset(transform=transform)

	# Test the dataloader
	dataloader = DataLoader(dataset=dataset,batch_size=batch_size)
	batch_sample = next(iter(dataloader))
	images, captions = batch_sample

	assert images.size() == (batch_size,) + (3,) + image_size
	assert len(captions) == batch_size

	for image in images:
		assert image.shape == (3,) + image_size 
