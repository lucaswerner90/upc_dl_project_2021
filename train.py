import torch
import os
from model.visualization import Visualization
from panel.main import tensorboard_panel
from torch.utils.data.dataset import Subset
import random
import numpy as np

def write_on_tensorboard(epoch:int, loss:int, bleu:int, image, expected_captions, generated_captions):
	tensorboard_panel.add_sentences_comparison(epoch,expected_captions[0],generated_captions[0])
	tensorboard_panel.add_loss(epoch,loss)
	tensorboard_panel.add_bleu(epoch,bleu)
	tensorboard_panel.add_image(epoch,image,expected_captions[0],generated_captions[0])

def split_subsets(dataset,train_percentage=0.8,all_captions=True):
	"""
	Performs the split of the dataset into Train and Test
	"""	
	if all_captions==True:

		# Get a list of all indexes in the dataset and convert to a numpy array  
		all_indexes = np.array([*range(0,len(dataset))])

		# Reshape the array so we can shuffle indexes in chunks of 5
		all_indexes_mat = all_indexes.reshape(-1,5)
		np.random.shuffle(all_indexes_mat)
		all_indexes_shuffled = all_indexes_mat.flatten()

		# Get the number of images for train and the rest are for test
		num_train_imgs = int(len(all_indexes_shuffled)/5*train_percentage)

		# Create the subsets for train and test
		train_split =  Subset(dataset,all_indexes_shuffled[0:num_train_imgs*5].tolist())
		test_split =  Subset(dataset,all_indexes_shuffled[num_train_imgs*5:].tolist())	

	else:
		all_first_index = [*range(0,len(dataset),5)]
		random.shuffle(all_first_index)

		num_train_imgs = int(len(all_first_index)*train_percentage)

		train_split =  Subset(dataset,all_first_index[0:num_train_imgs])
		test_split =  Subset(dataset,all_first_index[num_train_imgs:])	
		
	return train_split,test_split

def train_single_epoch(epoch, model, train_loader, optimizer, criterion, device,scheduler):
	"""
	Train single epoch
	"""
	model.train()
	for i, batch in enumerate(iter(train_loader)):
#	Si volem entrenar només amb un batch
# 		if i==0:
#			batch1 = batch
#			img, target = batch1
		img, target = batch
		img, target = img.to(device), target.to(device)

		optimizer.zero_grad()

		output = model(img, target)
		output = output.permute(1,2,0)
		loss = criterion(output[:,:,:-1], target[:,1:])    # target[:,1:])
		print(i, loss.item())
		loss.backward()

		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)

		optimizer.step()

#	Aixo és per fer servir el scheduer Exponential, que s'ha de fer estep cada cop que vulguis abaixar la gamma.
#		if (i+1)%10 == 0:
#			scheduler.step()
#		print(optimizer.param_groups[0]['lr'])

		candidate_corpus = [model.vocab.generate_caption(torch.argmax(output[0].transpose(1, 0), dim=-1))]
		reference_corpus = [model.vocab.generate_caption(target[0, 1:])]
		bleu = 0
		# bleu = bleu_score(candidate_corpus, reference_corpus)
		write_on_tensorboard(i+(epoch*len(train_loader)),loss.item(),bleu,img[0],reference_corpus,candidate_corpus)

def evaluate(model,test_loader, vocab, device,criterion):
	model.eval()

	total_loss = 0.
	#device= 'cpu'
	with torch.no_grad():
		for idx, batch in enumerate(iter(test_loader)):
			img, target = batch
			img = img.to(device)
			target = target.to(device)
			for i in range(img.shape[0]):
				sentence = model.inference(image=img[i].unsqueeze(0),vocab=vocab)
				alphas = model.forward(image=img[i].unsqueeze(0), vocab=vocab)[1]
			

			caption = ' '.join(sentence)
			Visualization.plot_attention((img[0]), sentence, alphas) # showing expected and plotting attention
			total_loss += target.numel()*criterion(sentence,target).item()
			n += target.numel()

		return total_loss / n, caption


def save_model(model, epoch):
	"""
	Function to save current model
	"""
	filename = os.path.join('model','checkpoints','Epoch_'+str(epoch)+'_model_state.pth')
	model_state = {
		'epoch':epoch,
		'model':model.state_dict()
	}
	torch.save(model_state, filename)

def train(num_epochs, model, train_loader,test_loader, optimizer, criterion, device,log_interval,vocab,scheduler):
	"""
	Executes model training. Saves model to a file every 5 epoch.
	"""	

	for epoch in range(1,num_epochs+1):
		train_single_epoch(epoch, model, train_loader,optimizer, criterion, device, scheduler)
		scheduler.step()
		if epoch % 5 == 0:
			save_model(model, epoch)
	
