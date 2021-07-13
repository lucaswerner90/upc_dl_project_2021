import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset.main import Flickr8kDataset
from dataset.caps_collate import CapsCollate
from dataset.download import DownloadDataset
from model.main import ImageCaptioningModel,ViTImageCaptioningModel
from train import train, split_subsets
from transformers import ViTFeatureExtractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

use_ViT_Enc = True

def main(args):

	if use_ViT_Enc:
		print("It is using ViT encoder!!!!")
		transform = None
		feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

	else:
		feature_extractor = None
		transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Resize((args['image_size'], args['image_size'])),
			# The normalize parameters depends on the model we're gonna use
			# If we apply transfer learning from a model that used ImageNet, then
			# we should use the ImageNet values to normalize the dataset.
			# Otherwise we could just normalize the values between -1 and 1 using the 
			# standard mean and standard deviation
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])

	dataset = Flickr8kDataset(dataset_folder='data', transform=transform,
								reduce=True, vocab_max_size=args['vocabulary_size'],feature_extractor=feature_extractor)

	# Create the model
	if use_ViT_Enc:
		model = ViTImageCaptioningModel(
			embed_size=args['embedding_dimension'],
			vocab = dataset.vocab,
			caption_max_length=args['captions_max_length'],
		).to(device)
	else:
		model = ImageCaptioningModel(
			image_features_dim=args['image_features_dimension'],
			embed_size=args['embedding_dimension'],
			vocab = dataset.vocab,
			caption_max_length=args['captions_max_length'],
		).to(device)

	# Perform the split of the dataset
	train_split, test_split = split_subsets(dataset,all_captions=True)

	train_loader = DataLoader(train_split, shuffle=True, batch_size=args['batch_size'], collate_fn=CapsCollate(
		pad_idx=dataset.vocab.word_to_index['<PAD>'], batch_first=True))

	test_loader = DataLoader(test_split, shuffle=True, batch_size=args['batch_size'], collate_fn=CapsCollate(
		pad_idx=dataset.vocab.word_to_index['<PAD>'], batch_first=True))

	optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'], betas=(0.9, 0.98), eps=1e-9)
	criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.word_to_index['<PAD>'])
	
	train(
		num_epochs=args['epochs'],
		model=model,
		train_loader=train_loader,
		test_loader=test_loader,
		optimizer=optimizer,
		criterion=criterion,
		device=device,
		log_interval=args['log_interval']
	)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Image captioning model setup')
	parser.add_argument('-bsz','--batch-size',type=int, required=False, choices=[4,8,16,32,64], default=64, help='Number of images to process on each batch')
	parser.add_argument('-vocab','--vocabulary-size',type=int, required=False, default=5000, help='Number of words that our model will use to generate the captions of the images')
	parser.add_argument('-image-feature','--image-features-dimension',type=int, choices=[256,512,1024], required=False, default=512, help='Number of features that the model will take for each image')
	parser.add_argument('-attn-dim','--attention-dimension',type=int, choices=[256,512,1024], required=False, default=256, help='Dimension of the attention tensor')
	parser.add_argument('-embed-dim','--embedding-dimension',type=int, choices=[256,512,1024], required=False, default=256, help='Dimension of the word embedding tensor')
	parser.add_argument('-epochs','--epochs',type=int, required=False, default=100, help='Number of epochs that our model will run')
	parser.add_argument('-captions-length','--captions-max-length',type=int, required=False, default=28, help='Max size of the predicted captions')
	parser.add_argument('-lr','--learning-rate',type=float, required=False, choices=[1e-1,1e-2,1e-3,1e-4],default=1e-3, help='Max size of the predicted captions')
	parser.add_argument('-img-size','--image-size',type=int, required=False, choices=[224,256,320], default=224, help='Size of the input image that our model will process')
	parser.add_argument('-log','--log-interval',type=int, required=False, default=5, help='During training, every X epochs, we log the results')

	args = parser.parse_args()
	variables = vars(args)

	if not os.path.exists('data'):
		print('Downloading Flickr8k dataset...')
		filepath = os.path.join(os.getcwd(),'data')
		DownloadDataset.download(filepath)

	main(variables)
