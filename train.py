from einops.einops import rearrange
import torch
from panel.main import tensorboard_panel
from torch.utils.data.dataset import Subset
from evaluate import evaluate, evaluate_tr
import random
import numpy as np


def write_on_tensorboard(epoch:int, model, loss:int, images, expected_captions, generated_captions):
	convert_caption = lambda caption: model.vocab.generate_phrase(caption)
	expected_captions = list(map(convert_caption, expected_captions))
	generated_captions = list(map(convert_caption, generated_captions))

	tensorboard_panel.add_sentences_comparison(epoch, expected_captions,generated_captions)
	#tensorboard_panel.add_images(epoch, images, expected_captions, generated_captions)
	tensorboard_panel.add_loss(epoch, loss)
	#tensorboard_panel.add_model_weights(epoch,model)
    	


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

		test_indexes = []
		for ind in all_first_index[num_train_imgs:]:
			for i in range(5):
				test_indexes.append(ind+i) 
		test_split =  Subset(dataset,test_indexes)	
		
	return train_split,test_split


def train_single_batch(epoch, model,batch,optimizer,criterion,device):
	img, target = batch
	img, target = img.to(device), target.to(device)

	optimizer.zero_grad()
	target[target==2]=0
	output = model(img,  target[:,:-1])
	output = rearrange(
		output,
		'bsz seq_len vocab_size -> bsz vocab_size seq_len'
	)
	loss = criterion(output, target[...,1:])
	print('--------------------------------------------------------------------------------------------------')
	print(f'Loss: {loss.item()}')
	
	loss.backward()
	torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25, error_if_nonfinite=True)
	optimizer.step()

	if not epoch%2:
		generated_captions = torch.argmax(output.transpose(1, 2), dim=-1)
		expected_captions = target[...,1:]
		generated_captions, expected_captions, images = generated_captions[:16,...], expected_captions[:16,...], img[:16,...]
		write_on_tensorboard(epoch=epoch, model=model, loss=loss.item(), images=images, expected_captions=expected_captions, generated_captions=generated_captions)


def train_single_epoch(epoch, model, train_loader, optimizer, criterion, device):
	"""
	Train single epoch
	"""
	for i, batch in enumerate(iter(train_loader)):
		img, target = batch
		img, target = img.to(device), target.to(device)

		optimizer.zero_grad()

		aux=torch.ones(target.shape[0],1,dtype=int)*model.vocab.word_to_index['<PAD>']
		aux=aux.to(target.device)
		target=torch.cat([target,aux],dim=1)

		target_loss=target
		
		output = model(img, target[:,:-1])
		output = rearrange(
			output,
			'bsz seq_len vocab_size -> bsz vocab_size seq_len'
		)
		loss = criterion(output, target_loss[:,1:])
		if i % 100 == 0:
			print('--------------------------------------------------------------------------------------------------')
			print(f'Epoch {epoch} batch: {i}/{len(train_loader)} loss: {loss.item()}')

		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)
		optimizer.step()

		generated_captions = torch.argmax(output.transpose(1, 2), dim=-1)
		expected_captions = target[...,1:]
		generated_captions, expected_captions, images = generated_captions[:16,...], expected_captions[:16,...], img[:16,...]
				
		write_on_tensorboard(epoch=len(train_loader)*(epoch-1)+i,model=model,loss=loss.item(),images=images,expected_captions=expected_captions,generated_captions=generated_captions)

def train(num_epochs, model, train_loader,test_loader, optimizer, criterion, device, log_interval):
	"""
	Executes model training. Saves model to a file every 5 epoch.
	"""	
	single_batch=False
	model.train()
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
	batch=next(iter(train_loader))
	for epoch in range(1,num_epochs+1):
		if single_batch:

			train_single_batch(epoch, model, batch,optimizer, criterion, device)

		else:
			train_single_epoch(epoch, model, train_loader,optimizer, criterion, device)
			scheduler.step()
			print(f'Scheduler: {scheduler.get_last_lr()[0]} ')
			if not epoch % log_interval:
				model.save_model(model, epoch)
			val_loss = evaluate_tr(model=model,test_loader=test_loader,device=device,epoch=epoch, criterion=criterion)
