import time
import torch
import math
import os
from model.visualization import Visualization
from torchtext.data.metrics import bleu_score
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

		print('-'*89)
		print(f'| epoch {epoch:3d} | {i:5d}/{len(train_loader):5d} batches | loss {total_loss:5.2f} | ppl {math.exp(total_loss):8.2f}')
		print('-'*89)
		print(f'Gen: {model.vocab.generate_caption(torch.argmax(output[0].transpose(1,0), dim=-1))}')
		print(f'Exp: {model.vocab.generate_caption(target[0,1:])}')
		print('-'*89)
		candidate_corpus = [model.vocab.generate_caption(torch.argmax(output[0].transpose(1, 0), dim=-1))]
		reference_corpus = [model.vocab.generate_caption(target[0, 1:])]
		print(f'Bleu score: {bleu_score(candidate_corpus, reference_corpus)}')
		total_loss = 0.

def evaluate(model,test_loader, vocab, device,criterion):#TODO:add device
	model.eval()

	total_loss = 0.
	sentences = []
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

def train(num_epochs, model, train_loader,test_loader, optimizer, criterion, device,log_interval):
	"""
	Executes model training. Saves model to a file every epoch.
	"""	

	for epoch in range(1,num_epochs+1):

		train_single_epoch(epoch, model, train_loader,optimizer, criterion, device, log_interval)

		if epoch % 5 == 0:
			save_model(model, epoch)
		
		# val_loss, _ = evaluate(model,test_loader,vocab,device,criterion)

		# print('-' * 89)
		# print(f'| end of epoch {epoch} | time: {(time.time() - epoch_start_time):.2f}s | valid loss {val_loss:.2f}')
		# print('-' * 89)

		
