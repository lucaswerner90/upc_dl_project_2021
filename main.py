import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataset import Flickr8kDataset
from caps_collate import CapsCollate

def main():
	data_path = "./dataset"
	#Setting hyperparameters
	hparams = {
		BATCH_SIZE = 256,
		#NUM_WORKER = 4,
		VOCAB_SIZE = 5000,
		reduce = True,
		NUM_EPOCHS = 10,
		max_length = 25,
		embed_size = 300,
		attention_dim = 256,
		encoder_dim = 2048,
		decoder_dim = 512,
		learning_rate = 1e-3,
		image_size = (128,128),
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	}
	#Choosing image transformations
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
	#Initializing dataset
    dataset = Flickr8kDataset(dataset_folder=data_path,transform=transform,reduce=True,max_size=VOCAB_SIZE)


if __name__ == "__main__":
	print(
	"""
	This script is gonna execute the whole program.
	That means, creating the dataset, training the model and executing the model
	"""
	)