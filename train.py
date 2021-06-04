import time
import torch
import math
import os

from evaluate import evaluate

def train_single_epoch(epoch, model, train_loader, optimizer, criterion, device, log_interval):
	"""
	Train single epoch
	"""
	#device = 'cpu'
	model.train()
	total_loss=0.
	for i, batch in enumerate(iter(train_loader)):
		batch_start_time = time.time()
		img, target = batch
		img, target = img.to(device), target.to(device)
		
		optimizer.zero_grad()

		#TODO: introduce for loop to make sentence
		output, _ = model(img, target)
		loss = criterion(output, target[:,1:])
		loss.backward()

		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)

		optimizer.step()

		total_loss += loss.item()

		if i % 10==0:
			print('-'*89)
			print(f'| epoch {epoch:3d} | {i:5d}/{len(train_loader):5d} batches | loss {total_loss:5.2f} | ppl {math.exp(total_loss):8.2f}')
			print('-'*89)
			print(f'Gen: {model.vocab.generate_caption(torch.argmax(output[0].transpose(1,0), dim=-1))}')
			print(f'Exp: {model.vocab.generate_caption(target[0,1:])}')
			print('-'*89)
		total_loss = 0.


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

def train(num_epochs, model, train_loader,test_loader, optimizer, criterion, device,log_interval,vocab):
	"""
	Executes model training. Saves model to a file every 5 epoch.
	"""	

	for epoch in range(1,num_epochs+1):
		epoch_start_time=time.time()
		train_single_epoch(epoch=epoch,model=model,train_loader=train_loader,optimizer=optimizer, criterion=criterion,device= device,log_interval=log_interval)

		if epoch % 5 == 0:
			save_model(model, epoch)
		
		val_loss = evaluate(model=model,test_loader=test_loader,vocab=vocab,device=device,epoch=epoch)

		print('-' * 89)
		print(f'| end of epoch {epoch} | time: {(time.time() - epoch_start_time):.2f}s | valid loss {val_loss:.2f}')
		print('-' * 89)



		
