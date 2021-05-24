import time
import torch
import math

def train_single_epoch(epoch, model, train_loader, optimizer, criterion, device, log_interval):
	"""
	Train single epoch
	"""
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

		if i % log_interval == 0 and i > 0:
			cur_loss = total_loss / log_interval
			elapsed = time.time() - batch_start_time
			print('-'*89)
			print(f'| epoch {epoch:3d} | {i:5d}/{len(train_loader):5d} batches | ms/batch {elapsed * 1000 / log_interval:5.2f} | loss {cur_loss:5.2f} | ppl {math.exp(cur_loss):8.2f}')
			print('-'*89)
			print(f'Generated: {torch.argmax(output[0].transpose(1,0), dim=-1)}')
			print(f'Expected: {target[0,1:]}')
			print('-'*89)
			total_loss = 0.

def evaluate(model,test_loader):
	model.eval()

	total_loss = 0.
	device= 'cpu'
	with torch.no_grad():
		for idx, batch in enumerate(iter(test_loader)):
			img, target = batch
			img = img.to(device)
			target = target.to(device)
			#TODO: Adapt this piece of code to Encoder and decoder implementation
			caption = ' '.join(output)

			total_loss += target.numel()*criterion(output,target).item()
			n += target.numel()

		return total_loss / n, caption


def save_model(model, epoch):
	"""
	Function to save current model
	"""
	model_state = {
		'epoch':epoch,
		'state_dict':model.state_dict()
	}
	torch.save(model_state,'Epoch_'+str(epoch)+'_model_state.pth')

def train(num_epochs, model, train_loader,test_loader, optimizer, criterion, device,log_interval):
	"""
	Executes model training. Saves model to a file every epoch.
	"""	
	epoch_start_time = time.time()

	for epoch in range(1,num_epochs+1):

		train_single_epoch(epoch, model, train_loader,optimizer, criterion, device, log_interval)

		val_loss, _ = evaluate(test_loader)

		print('-' * 89)
		print(f'| end of epoch {epoch} | time: {(time.time() - epoch_start_time):.2f}s | valid loss {val_loss:.2f}')
		print('-' * 89)

		save_model(model, epoch)
